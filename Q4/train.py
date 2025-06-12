# train.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import argparse
import json
import time
import logging

from event_driven_env import L2EventDrivenOrderBookEnv
from data_processor import L2BookDataProcessor
from ppo_agent import PPOAgent
from utils import plot_training_metrics # Import the updated plotting function

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a PPO agent on L2 order book data')
    parser.add_argument('--data_path', type=str, required=True, help='Path to line-delimited JSON L2 data file')
    parser.add_argument('--levels', type=int, default=10)
    parser.add_argument('--history_length', type=int, default=20)
    parser.add_argument('--init_cash', type=float, default=10000.0)
    parser.add_argument('--max_position', type=int, default=50)
    parser.add_argument('--maker_fee', type=float, default=-0.0002)
    parser.add_argument('--taker_fee', type=float, default=0.0007)
    parser.add_argument('--inventory_penalty', type=float, default=0.05)
    parser.add_argument('--taker_penalty', type=float, default=0.01, help='Penalty for aggressive market orders.')
    parser.add_argument('--maker_reward', type=float, default=0.01, help='Reward for passive maker fills.')
    parser.add_argument('--pnl_reward_multiplier', type=float, default=5.0, help='Multiplier for realized PnL reward component.')
    parser.add_argument('--episode_length_seconds', type=int, default=1800)
    parser.add_argument('--num_episodes', type=int, default=250)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_freq', type=int, default=50, help='Save model and plots every N episodes')
    return parser.parse_args()

def create_environment(args, data_processor):
    return L2EventDrivenOrderBookEnv(
        data_processor=data_processor, levels=args.levels, history_length=args.history_length,
        init_cash=args.init_cash, max_position=args.max_position, maker_fee=args.maker_fee,
        taker_fee=args.taker_fee, inventory_penalty=args.inventory_penalty,        taker_penalty=args.taker_penalty,
        maker_reward=args.maker_reward,
        pnl_reward_multiplier=args.pnl_reward_multiplier,
        episode_length_seconds=args.episode_length_seconds, seed=args.seed
    )

def create_agent(args, env):
    return PPOAgent(
        state_dim=env.observation_space.shape, action_space=env.action_space,
        hidden_size=128, lstm_units=64, learning_rate=args.learning_rate, epsilon=0.2,
        entropy_coef=0.01, value_coef=0.5, gamma=0.99, lam=0.95,
        buffer_size=args.buffer_size, batch_size=args.batch_size, update_epochs=10, max_grad_norm=0.5
    )

def train(args):
    # Setup directories and seeds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"l2_ppo_training_{timestamp}")
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_processor = L2BookDataProcessor(args.data_path)
    env = create_environment(args, data_processor)
    agent = create_agent(args, env)
    
    # --- MODIFIED: More comprehensive tracking ---
    all_episode_rewards = []
    training_episode_metrics = []
    
    training_start_time = time.time()
    
    for episode in range(1, args.num_episodes + 1):
        state, info = env.reset(seed=args.seed + episode, options={'random_start': True})
        if state is None:
            logging.warning(f"Failed to reset environment for episode {episode}, likely at end of data. Stopping training.")
            break

        episode_reward = 0
        done = False
        
        # --- CORRECTED LEARNING LOGIC ---
        # The agent collects experience until the episode is done OR the buffer is full.
        # Then, it learns from the collected experience.
        
        while not done:
            action, value, log_prob = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            agent.store_transition(state, action, reward, value, log_prob, done)
            state = next_state
            episode_reward += reward
            
            # If buffer is full, it's time to learn, even if episode isn't over.
            if agent.buffer.ptr >= agent.buffer_size:
                last_val = agent.get_value(state) if not done else 0.0
                agent.finish_path(last_val)
                loss_info = agent.learn() # This also resets the buffer
                if loss_info:
                    logging.info(f"Ep {episode} | Mid-episode policy update | Actor Loss: {loss_info['actor_loss']:.4f}")
        
        # If episode ends before buffer is full, learn from the partial data.
        if not agent.buffer.is_empty():
            agent.finish_path(last_value=0.0) # Episode is done, so bootstrap is 0
            agent.learn()
        
        # Log episode results
        all_episode_rewards.append(episode_reward)
        training_episode_metrics.append(info) # Store final info dict
        
        pnl = info.get('pnl', 0.0)
        avg_reward = np.mean(all_episode_rewards[-50:])
        logging.info(f"Episode {episode}/{args.num_episodes} | Reward: {episode_reward:.2f} | PnL: ${pnl:.2f} | Fill Ratio: {info.get('fill_ratio', 0):.2f} | Avg Reward (last 50): {avg_reward:.2f}")

        # Save models and plots periodically
        if episode % args.save_freq == 0:
            agent.save(os.path.join(model_dir, f'model_ep{episode}'))
            plot_training_metrics(
                episode_rewards=all_episode_rewards, 
                training_episode_metrics=training_episode_metrics, # Pass detailed metrics
                output_path=os.path.join(output_dir, 'training_curves.png')
            )
            
    # --- FINAL SAVING ---
    logging.info("Training finished. Saving final model and results.")
    agent.save(os.path.join(model_dir, 'final_model'))
    
    # Save final plots and metrics data
    plot_training_metrics(
        episode_rewards=all_episode_rewards,
        training_episode_metrics=training_episode_metrics,
        output_path=os.path.join(output_dir, 'final_training_curves.png')
    )
    
    # Save detailed training metrics to a CSV for later analysis
    metrics_df = pd.DataFrame(training_episode_metrics)
    metrics_df['episode'] = range(1, len(metrics_df) + 1)
    metrics_df.to_csv(os.path.join(output_dir, 'training_summary_metrics.csv'), index=False)

    total_time = time.time() - training_start_time
    logging.info(f"Total training time: {total_time/3600:.2f} hours.")

if __name__ == "__main__":
    args = parse_args()
    train(args)