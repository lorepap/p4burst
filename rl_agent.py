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
import importlib.util

# Import FEATURE_NAMES from create_rl_dataset.py
def import_feature_names():
    """Import FEATURE_NAMES from create_rl_dataset.py"""
    spec = importlib.util.spec_from_file_location(
        "create_rl_dataset", "/home/ubuntu/p4burst/create_rl_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FEATURE_NAMES

# Get feature names from create_rl_dataset.py
FEATURE_NAMES = import_feature_names()

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=32, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma  # discount factor
        self.batch_size = batch_size
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def update_target_model(self):
        # Copy weights to target model
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, state, action, reward, next_state, done):
        # Ensure state and next_state have consistent shapes
        if hasattr(state, 'shape') and len(state.shape) > 1:
            state = state.flatten()
        if hasattr(next_state, 'shape') and len(next_state.shape) > 1:
            next_state = next_state.flatten()
            
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Always choose the best action without exploration
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample from memory but don't convert to numpy array yet
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract components separately
        states = np.array([experience[0] for experience in minibatch], dtype=np.float32)
        actions = np.array([experience[1] for experience in minibatch], dtype=np.int32)
        rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
        next_states = np.array([experience[3] for experience in minibatch], dtype=np.float32)
        dones = np.array([experience[4] for experience in minibatch], dtype=np.float32)

        # Q(s', a)
        # Use 1-dones instead of ~dones (bitwise not doesn't work with TensorFlow)
        target = rewards + self.gamma * np.max(self.target_model.predict(next_states, verbose=0), axis=1) * (1-dones)
        target_f = self.model.predict(states, verbose=0)
        
        # Q(s, a)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        # Train the neural network
        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        
        return history.history['loss'][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def preprocess_dataset(csv_file):
    """
    Preprocess the dataset for offline DQN training.
    Uses the FEATURE_NAMES list from create_rl_dataset.py for consistency.
    """
    data = pd.read_csv(csv_file)
    
    print("\nDataset sample:")
    print(data.head())
    print(f"\nAction distribution: {data['action'].value_counts().to_dict()}")
    print(f"Reward statistics: Min={data['reward'].min()}, Max={data['reward'].max()}, Mean={data['reward'].mean():.2f}")
    
    # Use the imported FEATURE_NAMES for state representation
    feature_cols = list(FEATURE_NAMES)  # Make a copy to avoid modifying the original
    
    # Check which features are actually available in the dataset
    available_features = [col for col in feature_cols if col in data.columns]
    missing_features = [col for col in feature_cols if col not in data.columns]
    
    if missing_features:
        print(f"\nWarning: Some expected features are missing from the dataset: {missing_features}")
    
    print(f"\nUsing {len(available_features)} state features: {available_features}")
    
    # Normalize state features
    data_norm = data.copy()
    
    # Normalize queue depth features
    queue_depth_cols = [col for col in available_features if col.startswith('queue_depth_')]
    for col in queue_depth_cols:
        max_val = data[col].max()
        if max_val > 0:  # Avoid division by zero
            data_norm[col] = data[col] / max_val
            print(f"Normalized {col}: max value = {max_val}")
    
    if 'packet_size' in available_features:
        max_pkt_size = data['packet_size'].max()
        if max_pkt_size > 0:
            data_norm['packet_size'] = (data['packet_size'] / max_pkt_size).round(2)
            print(f"Normalized packet_size: max value = {max_pkt_size}")
    
    # Extract state, action, reward
    states = data_norm[available_features].values
    actions = data_norm['action'].values
    rewards = data_norm['reward'].values
    
    # Create next_state by shifting
    next_states = np.vstack([states[1:], states[-1]])
    
    # Create done flag (True for last state, False otherwise)
    dones = np.zeros(len(states), dtype=bool)
    dones[-1] = True
    
    return states, actions, rewards, next_states, dones, available_features


def train_offline_dqn(csv_file, model_dir, epochs=100, batch_size=32):
    """
    Train a DQN agent using offline data.
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and preprocess the dataset
    states, actions, rewards, next_states, dones, _ = preprocess_dataset(csv_file)
    
    # Get state and action dimensions
    state_size = states.shape[1]
    action_size = len(np.unique(actions))
    
    print(f"\nState size: {state_size}, Action size: {action_size}")
    print(f"Dataset size: {len(states)} samples")
    
    # Print some example state-action-reward triples
    print("\nExample state-action-reward samples:")
    for i in range(min(5, len(states))):
        print(f"State {i}: {states[i]}")
        print(f"Action: {actions[i]}")
        print(f"Reward: {rewards[i]}")
        print(f"Next state: {next_states[i]}")
        print("---")
    
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
        
        #if e % 10 == 0:
        print(f"Epoch {e}/{epochs}, Loss: {loss:.4f}")
        
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
    
    print(f"\nTraining completed. Model saved to {os.path.join(model_dir, 'dqn_model.h5')}")
    
    return agent


def evaluate_agent(agent, csv_file):
    """
    Evaluate the trained agent based on cumulative reward.
    """
    # Load and preprocess the dataset
    states, actions, rewards, next_states, dones, _ = preprocess_dataset(csv_file)

    # Track agent's predictions
    predicted_actions = []
    total_reward = 0
    
    print("\nEvaluating agent on test set...")
    for i in range(len(states)):
        state = states[i].reshape(1, -1)
        action = agent.act(state)
        predicted_actions.append(action)
        total_reward += rewards[i]
    
    # Compare predictions with dataset actions
    actions_match = np.array(predicted_actions) == actions
    match_percentage = 100 * np.mean(actions_match)
    
    print(f"\nAction match percentage: {match_percentage:.2f}%")
    print(f"Predicted action distribution: {np.unique(predicted_actions, return_counts=True)}")
    print(f"Actual action distribution: {np.unique(actions, return_counts=True)}")
    
    avg_reward = total_reward / len(states)
    print(f"Agent evaluation: Average reward = {avg_reward:.4f}")
    
    return avg_reward

def evaluate_network_performance(agent, csv_file):
    """
    Evaluate agent based on network performance metrics like queue depth.
    """
    states, _, rewards, next_states, dones, _ = preprocess_dataset(csv_file)

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
    parser.add_argument('--model_dir', default='model', help='Directory to save the model')
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
    #evaluate_agent(agent, args.csv)


if __name__ == '__main__':
    main()