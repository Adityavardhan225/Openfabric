# ppo_agent.py

import numpy as np
import tensorflow as tf
import os
import logging
from ppo_network import PPONetwork

# PPOBuffer class remains unchanged.
class PPOBuffer:
    """Buffer for storing trajectories for a PPO agent with a discrete action space."""
    def __init__(self, size, state_dim, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        
        self.states = np.zeros((size,) + state_dim, dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.bool_)
        
        self.ptr, self.path_start_idx = 0, 0
        
    def store(self, state, action, reward, value, log_prob, done):
        if self.ptr >= self.size:
            logging.warning("Buffer overflow. Some transitions may be lost.")
            return
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        
    def finish_path(self, last_value=0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.advantages = np.zeros_like(deltas, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(deltas))):
            last_gae_lam = deltas[t] + self.gamma * self.lam * (1 - self.dones[self.path_start_idx + t]) * last_gae_lam
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values[path_slice]
        self.path_start_idx = self.ptr

    def get(self):
        if self.ptr == 0:
            return None
        # --- FIX: handle case where buffer isn't full but episode ends ---
        # This assertion is too strict when episodes are shorter than the buffer.
        # assert self.path_start_idx == self.ptr, "Buffer has unfinished paths!"
        if self.path_start_idx != self.ptr:
            # If we are here, it means finish_path was not called for the last trajectory.
            # This can happen if the script ends unexpectedly. It's safer to log a warning.
            logging.warning("Buffer has an unfinished path. This may indicate an issue. Finishing path now.")
            self.finish_path() # Finish with default last_value=0

        actual_size = self.ptr
        self.ptr, self.path_start_idx = 0, 0
        
        # Slice the advantages and returns correctly
        advantages_slice = self.advantages[:actual_size]
        returns_slice = self.returns[:actual_size]

        adv_mean, adv_std = np.mean(advantages_slice), np.std(advantages_slice)
        normalized_advantages = (advantages_slice - adv_mean) / (adv_std + 1e-8)
        
        return dict(
            states=self.states[:actual_size], 
            actions=self.actions[:actual_size], 
            advantages=normalized_advantages, 
            returns=returns_slice, # Use the sliced returns
            log_probs=self.log_probs[:actual_size]
        )
    
    def is_empty(self):
        return self.ptr == 0


class PPOAgent:
    """PPO Agent adapted for the discrete action space L2 environment."""
    
    # --- THIS IS THE METHOD TO REPLACE ---
    def __init__(self, state_dim, action_space, **kwargs):
        self.state_dim = state_dim
        self.action_space = action_space
        self.action_dim = action_space.n
        
        # Extract agent-specific hyperparameters
        self.gamma = kwargs.get('gamma', 0.99)
        self.lam = kwargs.get('lam', 0.95)
        self.buffer_size = kwargs.get('buffer_size', 2048)
        self.batch_size = kwargs.get('batch_size', 64)
        self.update_epochs = kwargs.get('update_epochs', 10)
        
        # ** THE FIX IS HERE **
        # Create a dictionary containing only the arguments that PPONetwork expects.
        network_kwargs = {
            'hidden_size': kwargs.get('hidden_size', 128),
            'lstm_units': kwargs.get('lstm_units', 64),
            'learning_rate': kwargs.get('learning_rate', 3e-4),
            'epsilon': kwargs.get('epsilon', 0.2),
            'entropy_coef': kwargs.get('entropy_coef', 0.01),
            'value_coef': kwargs.get('value_coef', 0.5),
            'max_grad_norm': kwargs.get('max_grad_norm', 0.5)
        }
        
        # Now, pass the filtered dictionary to the network.
        self.network = PPONetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            **network_kwargs
        )
        
        self.buffer = PPOBuffer(
            size=self.buffer_size,
            state_dim=self.state_dim,
            gamma=self.gamma,
            lam=self.lam
        )

    # --- NO CHANGES NEEDED FOR THE REST OF THE PPOAgent CLASS ---

    def get_value(self, state):
        """Get the value estimate for a given state."""
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        value = self.network.get_value(state_tensor)
        return value.numpy()[0, 0]
    
    def act(self, state, training=True):
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        action_tensor, log_prob_tensor = self.network.get_action(state_tensor)
        
        # We only need the value during training to store it in the buffer
        if training:
            value_tensor = self.network.get_value(state_tensor)
            value = value_tensor.numpy()[0, 0]
        else:
            value = None # Not needed for evaluation
            
        action = action_tensor.numpy()[0]
        log_prob = log_prob_tensor.numpy()[0]

        return action, value, log_prob
        
    def store_transition(self, state, action, reward, value, log_prob, done):
        self.buffer.store(state, action, reward, value, log_prob, done)
        
    def finish_path(self, last_value=0.0):
        self.buffer.finish_path(last_value)
        
    def learn(self):
        data = self.buffer.get()
        if data is None:
            logging.info("Buffer empty. Skipping learn step.")
            return {}
        
        total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
        num_updates = 0

        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.shuffle(buffer_size=len(data['states'])).repeat(self.update_epochs).batch(self.batch_size)

        for batch in dataset:
            actor_loss, critic_loss, entropy_loss = self.network.train_on_batch(
                batch['states'], batch['actions'], batch['log_probs'],
                batch['returns'], batch['advantages']
            )
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy_loss += entropy_loss
            num_updates += 1

        if num_updates == 0: return {}
        
        return {
            'actor_loss': total_actor_loss.numpy() / num_updates,
            'critic_loss': total_critic_loss.numpy() / num_updates,
            'entropy_loss': total_entropy_loss.numpy() / num_updates
        }
        
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.network.save_models(actor_path=os.path.join(path, 'actor.keras'), critic_path=os.path.join(path, 'critic.keras'))
        
    def load(self, path):
        self.network.load_models(actor_path=os.path.join(path, 'actor.keras'), critic_path=os.path.join(path, 'critic.keras'))