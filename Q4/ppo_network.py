# ppo_network.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

class PPONetwork:
    """
    PPO Network with recurrent layers, adapted for a discrete action space.
    """
    def __init__(self, 
                 state_dim,
                 action_dim, # MODIFICATION: Now a simple integer dimension
                 hidden_size=128,
                 lstm_units=64,
                 learning_rate=3e-4,
                 epsilon=0.2,
                 entropy_coef=0.01,
                 value_coef=0.5,
                 max_grad_norm=0.5):
        
        self.state_dim = state_dim
        self.action_dim = action_dim # No more action_type_dim or action_param_dim
        self.hidden_size = hidden_size
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.actor, self.critic = self._build_networks()
        self.optimizer = Adam(learning_rate=self.learning_rate)
        
    def _build_networks(self):
        """Build actor and critic networks with shared features."""
        state_input = Input(shape=self.state_dim)
        
        # Feature extraction layers
        conv1 = Conv1D(32, kernel_size=3, activation='relu')(state_input)
        # Using LSTM to capture temporal patterns
        lstm_out = LSTM(self.lstm_units, return_sequences=False)(conv1)
        
        # Shared dense layer
        shared = Dense(self.hidden_size, activation='relu')(lstm_out)
        
        # --- MODIFICATION: Simplified Actor Head ---
        # The actor only needs to output probabilities for the discrete actions.
        action_probs = Dense(self.action_dim, activation='softmax', name='action_probs')(shared)
        
        # Critic head (value function) is unchanged
        value = Dense(1, name='value')(shared)
        
        # Create models
        actor = Model(inputs=state_input, outputs=action_probs)
        critic = Model(inputs=state_input, outputs=value)
        
        return actor, critic
        
    @tf.function # Decorator for performance
    def get_action(self, state):
        """Sample an action from the policy. Uses tf.function for speed."""
        action_probs = self.actor(state, training=False)
        dist = tfp.distributions.Categorical(probs=action_probs)
        
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob

    @tf.function # Decorator for performance
    def get_value(self, state):
        """Get value estimate for a state."""
        return self.critic(state, training=False)

    def evaluate_actions(self, states, actions):
        """Evaluate log probability and entropy of given actions."""
        action_probs = self.actor(states, training=False)
        dist = tfp.distributions.Categorical(probs=action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy
    
    @tf.function
    def train_on_batch(self, states, actions, old_log_probs, returns, advantages):
        """Train the PPO model on a batch of data using GradientTape."""
        with tf.GradientTape() as tape:
            # Actor loss
            new_log_probs, entropy = self.evaluate_actions(states, actions)
            ratio = tf.exp(new_log_probs - old_log_probs)
            
            surrogate1 = ratio * advantages
            surrogate2 = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
            
            # Critic loss
            values = self.critic(states, training=True)
            values = tf.squeeze(values, axis=1)
            critic_loss = tf.reduce_mean(tf.square(returns - values))
            
            # Entropy bonus to encourage exploration
            entropy_loss = -tf.reduce_mean(entropy)
            
            # Total loss
            total_loss = actor_loss + (self.value_coef * critic_loss) + (self.entropy_coef * entropy_loss)
            
        # Calculate and apply gradients
        trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
        grads = tape.gradient(total_loss, trainable_vars)
        
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return actor_loss, critic_loss, entropy_loss

    def save_models(self, actor_path, critic_path):
        self.actor.save(actor_path)
        self.critic.save(critic_path)
        
    def load_models(self, actor_path, critic_path):
        self.actor = tf.keras.models.load_model(actor_path)
        self.critic = tf.keras.models.load_model(critic_path)