# evaluate.py

import os
import numpy as np
import pandas as pd
import argparse
import json
import time
from datetime import datetime
import logging
from typing import Dict, List

from event_driven_env import L2EventDrivenOrderBookEnv
from data_processor import L2BookDataProcessor
from ppo_agent import PPOAgent
from utils import calculate_sharpe_ratio, calculate_sortino_ratio, plot_trades_and_pnl, plot_evaluation_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained PPO agent on L2 order book data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model directory (e.g., results/run_xyz/models/final_model)')
    parser.add_argument('--data_path', type=str, required=True, help='Path to line-delimited JSON L2 data file')
    parser.add_argument('--params_file', type=str, help='Path to JSON file with training parameters (auto-detected if not provided)')
    parser.add_argument('--num_episodes', type=int, default=10, help='Number of episodes to evaluate for statistical significance')
    parser.add_argument('--visualize_first', action='store_true', help='Generate detailed trade visualization for the first episode')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed for evaluation')
    return parser.parse_args()

def load_parameters(args):
    params_path = args.params_file or os.path.join(os.path.dirname(os.path.dirname(args.model_path)), 'params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
            logging.info(f"Loaded parameters from {params_path}")
            return params
    else:
        logging.warning("Parameter file not found. Using default evaluation parameters.")
        return {'levels': 10, 'history_length': 20, 'init_cash': 10000.0, 'max_position': 50, 'maker_fee': -0.0002, 'taker_fee': 0.0007, 'inventory_penalty': 0.01, 'episode_length_seconds': 1800}

def create_environment(params, data_processor):
    return L2EventDrivenOrderBookEnv(
        data_processor=data_processor, levels=params.get('levels', 10), history_length=params.get('history_length', 20),
        init_cash=params.get('init_cash', 10000.0), max_position=params.get('max_position', 50),
        maker_fee=params.get('maker_fee', -0.0002), taker_fee=params.get('taker_fee', 0.0007),
        inventory_penalty=params.get('inventory_penalty', 0.01), episode_length_seconds=params.get('episode_length_seconds', 1800)
    )

def create_agent(params, env):
    return PPOAgent(
        state_dim=env.observation_space.shape, action_space=env.action_space,
        hidden_size=params.get('hidden_size', 128), lstm_units=params.get('lstm_units', 64)
    )

def evaluate_episode(env, agent, episode_seed):
    state, info = env.reset(seed=episode_seed, options={'random_start': True})
    if state is None: return None, None, None

    pnl_history, inventory_history, mid_price_history = [], [], []
    
    start_time = time.time()
    
    while True:
        action, _, _ = agent.act(state, training=False)
        next_state, reward, done, _, info = env.step(action)
        
        pnl_history.append((info['timestamp'], info['pnl']))
        inventory_history.append(info['inventory'])
        mid_price_history.append(info['mid_price'])
        
        state = next_state
        if done: break
            
    elapsed_time = time.time() - start_time
    
    final_pnl_values = [p for ts, p in pnl_history]
    sharpe = calculate_sharpe_ratio(final_pnl_values)
    sortino = calculate_sortino_ratio(final_pnl_values)
    
    peak = np.maximum.accumulate(final_pnl_values)
    drawdown = peak - final_pnl_values
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # --- MODIFIED: Ensure all key metrics are captured ---
    metrics = {
        'final_pnl': info.get('pnl', 0.0),
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'fill_ratio': info.get('fill_ratio', 0.0),
        'adverse_selection_ratio': info.get('adverse_selection_ratio', 0.0),
        'maker_volume': info.get('maker_volume', 0.0),
        'taker_volume': info.get('taker_volume', 0.0),
        'fees_paid': info.get('fees_paid', 0.0),
        'episode_duration_sec': elapsed_time
    }
    
    history = {'pnl': pnl_history, 'inventory': inventory_history, 'mid_price': mid_price_history}
    
    return metrics, history, env.agent_trades

def run_evaluation(args):
    np.random.seed(args.seed)
    output_dir = os.path.join(args.output_dir, f"eval_{os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    
    params = load_parameters(args)
    with open(os.path.join(output_dir, 'eval_params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    data_processor = L2BookDataProcessor(args.data_path)
    env = create_environment(params, data_processor)
    agent = create_agent(params, env)
    agent.load(args.model_path)
    logging.info(f"Loaded trained model from {args.model_path}")

    all_metrics = []
    for episode in range(args.num_episodes):
        logging.info(f"--- Starting evaluation episode {episode+1}/{args.num_episodes} ---")
        metrics, history, trades = evaluate_episode(env, agent, episode_seed=args.seed + episode)
        
        if metrics is None:
            logging.warning("Evaluation ended prematurely, possibly due to end of data.")
            break
        
        all_metrics.append(metrics)
        logging.info(f"Episode {episode+1} Results: PnL=${metrics['final_pnl']:.2f}, Fill Ratio={metrics['fill_ratio']:.2f}, Adverse Selection={metrics['adverse_selection_ratio']:.3f}")

        if args.visualize_first and episode == 0:
            plot_trades_and_pnl(
                pnl_history=history['pnl'], inventory_history=history['inventory'],
                mid_price_history=history['mid_price'], trades=trades,
                output_path=os.path.join(output_dir, 'episode_1_visualization.png')
            )

    if not all_metrics:
        logging.error("No episodes were completed. Evaluation failed.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'evaluation_summary.csv'), index=False)
    
    # --- ADDED: Call to summary plot function ---
    plot_evaluation_summary(metrics_df, output_path=os.path.join(output_dir, 'evaluation_summary_plot.png'))

    logging.info("\n" + "="*50)
    logging.info(" " * 10 + "Average Evaluation Results")
    logging.info("="*50)
    # Pretty print the mean results
    for key, value in metrics_df.mean().items():
        logging.info(f"{key:<25}: {value:.4f}")
    logging.info("="*50)

if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)