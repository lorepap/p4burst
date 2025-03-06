#!/usr/bin/env python3
# filepath: /home/ubuntu/p4burst/train_rl_agent.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import argparse
import matplotlib.pyplot as plt
import random
import os

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=32, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential([
            keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target_model(self):
        # Copy weights to target model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = np.array(random.sample(self.memory, self.batch_size))
        states = np.vstack(minibatch[:, 0])
        actions = minibatch[:, 1].astype(np.int32)
        rewards = minibatch[:, 2].astype(np.float32)
        next_states = np.vstack(minibatch[:, 3])
        dones = minibatch[:, 4].astype(np.bool_)

        # Q(s', a)
        target = rewards + self.gamma * np.max(self.target_model.predict(next_states, verbose=0), axis=1) * ~dones
        target_f = self.model.predict(states, verbose=0)
        
        # Q(s, a)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        # Train the neural network
        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return history.history['loss'][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def preprocess_dataset(csv_file):
    """
    Preprocess the dataset for offline DQN training.
    """
    data = pd.read_csv(csv_file)
    
    # Extract features (state variables)
    feature_cols = [col for col in data.columns if col.startswith('queue_depth_')]
    
    # Normalize state features
    data_norm = data.copy()
    for col in feature_cols:
        max_val = data[col].max()
        if max_val > 0:  # Avoid division by zero
            data_norm[col] = data[col] / max_val
            
    # Extract state, action, reward
    states = data_norm[feature_cols].values
    actions = data_norm['action'].values
    rewards = data_norm['reward'].values
    
    # Create next_state by shifting
    next_states = np.vstack([states[1:], states[-1]])
    
    # Create done flag (True for last state, False otherwise)
    dones = np.zeros(len(states), dtype=bool)
    dones[-1] = True
    
    return states, actions, rewards, next_states, dones


def train_offline_dqn(csv_file, model_dir, epochs=100, batch_size=32):
    """
    Train a DQN agent using offline data.
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess the dataset
    states, actions, rewards, next_states, dones = preprocess_dataset(csv_file)
    
    # Get state and action dimensions
    state_size = states.shape[1]
    action_size = len(np.unique(actions))
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Dataset size: {len(states)} samples")
    
    # Initialize the agent
    agent = DQNAgent(state_size, action_size, batch_size=batch_size)
    
    # Fill the agent's memory with the dataset
    for i in range(len(states)):
        agent.memorize(states[i].reshape(1, -1), actions[i], rewards[i], 
                      next_states[i].reshape(1, -1), dones[i])
    
    # Training loop
    losses = []
    for e in range(epochs):
        loss = agent.replay()
        losses.append(loss)
        
        if e % 10 == 0:
            print(f"Epoch {e}/{epochs}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}")
        
        # Update target model periodically
        if e % 50 == 0:
            agent.update_target_model()
    
    # Save the trained model
    agent.save(os.path.join(model_dir, "dqn_model.h5"))
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(model_dir, 'training_loss.png'))
    
    print(f"Training completed. Model saved to {os.path.join(model_dir, 'dqn_model.h5')}")
    
    return agent


def evaluate_agent(agent, csv_file):
    """
    Evaluate the trained agent based on cumulative reward.
    """
    # Load and preprocess the dataset
    states, _, rewards, next_states, dones = preprocess_dataset(csv_file)

    total_reward = 0
    for i in range(len(states)):
        action = agent.act(states[i].reshape(1, -1))  # Agent chooses action
        total_reward += rewards[i]  # Sum rewards (assumes rewards are aligned with states)
    
    avg_reward = total_reward / len(states)
    print(f"Agent evaluation: Average reward = {avg_reward:.4f}")
    
    return avg_reward

def evaluate_network_performance(agent, csv_file):
    """
    Evaluate agent based on network performance metrics like queue depth.
    """
    states, _, rewards, next_states, dones = preprocess_dataset(csv_file)

    total_queue_depth_reduction = 0
    for i in range(len(states)):
        action = agent.act(states[i].reshape(1, -1))
        total_queue_depth_reduction += states[i].sum() - next_states[i].sum()

    avg_queue_reduction = total_queue_depth_reduction / len(states)
    print(f"Average Queue Depth Reduction: {avg_queue_reduction:.4f}")

    return avg_queue_reduction



def main():
    parser = argparse.ArgumentParser(description='Train offline DQN agent')
    parser.add_argument('--csv', required=True, help='Path to the dataset CSV')
    parser.add_argument('--model_dir', default='trained_models', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    import random
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Train the agent
    agent = train_offline_dqn(args.csv, args.model_dir, epochs=args.epochs, batch_size=args.batch_size)
    
    # Evaluate the agent
    evaluate_agent(agent, args.csv)


if __name__ == '__main__':
    main()