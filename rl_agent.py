#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque, defaultdict
import argparse
import matplotlib.pyplot as plt
import random
import os
import importlib.util
import json
import glob

# Import FEATURE_NAMES from create_rl_dataset.py
def import_feature_names():
    """Import FEATURE_NAMES from create_rl_dataset.py"""
    spec = importlib.util.spec_from_file_location(
        "create_rl_dataset", "/home/ubuntu/p4burst/create_rl_dataset.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FEATURE_NAMES

# If FEATURE_NAMES doesn't include fw_port_depth, we need to add it
# And ensure switch_id is removed
FEATURE_NAMES = import_feature_names()
if 'fw_port_depth' not in FEATURE_NAMES:
    FEATURE_NAMES.append('fw_port_depth')
if 'switch_id' in FEATURE_NAMES:
    FEATURE_NAMES.remove('switch_id')

def discover_experiment_datasets(specific_folder=None):
    """
    Discover experiment datasets in the workspace.
    If specific_folder is provided, look for final_dataset.csv in that folder only.
    Otherwise, automatically discover all experiment datasets.
    
    Args:
        specific_folder (str, optional): Path to a specific experiment folder.
        
    Returns:
        list: Paths to all final_dataset.csv files found.
    """
    dataset_paths = []
    
    # If a specific folder is provided, use that instead of auto-discovery
    if specific_folder:
        dataset_path = os.path.join(specific_folder, "final_dataset.csv")
        if os.path.exists(dataset_path):
            dataset_paths.append(dataset_path)
            print(f"Using provided dataset: {dataset_path}")
        else:
            print(f"Warning: No final_dataset.csv found in {specific_folder}")
        return dataset_paths
        
    # Automatic discovery (existing functionality)
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")
    
    print(f"Looking for experiment directories in: {base_path}")
    
    # Find all directories matching experiment_* pattern
    all_dirs = sorted(glob.glob(os.path.join(base_path, "experiment_*")))
    
    # Filter to only include directories where the last character is a digit
    experiment_dirs = [d for d in all_dirs if os.path.basename(d)[-1].isdigit()]
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    print(f"Filtered out {len(all_dirs) - len(experiment_dirs)} non-experiment directories")
    
    # Find final_dataset.csv in each experiment directory
    for exp_dir in experiment_dirs:
        dataset_path = os.path.join(exp_dir, "final_dataset.csv")
        if os.path.exists(dataset_path):
            dataset_paths.append(dataset_path)
            print(f"Found dataset: {dataset_path}")
        else:
            print(f"Warning: No final_dataset.csv found in {exp_dir}")
    
    return dataset_paths

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
    
    def memorize_batch(self, states, actions, rewards, next_states, dones):
        """Add multiple experiences to memory at once"""
        for i in range(len(states)):
            self.memorize(states[i], actions[i], rewards[i], next_states[i], dones[i])

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


def get_global_normalization_parameters(csv_files):
    """
    Calculate global normalization parameters across all datasets.
    Returns a dictionary of scale factors for each feature.
    """
    scale_factors = {}
    
    print("Calculating global normalization parameters...")
    for csv_file in csv_files:
        print(f"Processing {csv_file} for normalization...")
        data = pd.read_csv(csv_file)
        
        # Get available features - ensure we don't look for switch_id
        feature_cols = [col for col in FEATURE_NAMES if col in data.columns and col != 'switch_id']
        
        # Find max value for total queue depth and fw_port_depth
        if 'total_queue_depth' in feature_cols:
            max_val = data['total_queue_depth'].max()
            if 'total_queue_depth' in scale_factors:
                scale_factors['total_queue_depth'] = max(scale_factors['total_queue_depth'], int(max_val))
            else:
                scale_factors['total_queue_depth'] = int(max_val)
                
        # Add normalization for fw_port_depth
        if 'fw_port_depth' in feature_cols:
            max_val = data['fw_port_depth'].max()
            if 'fw_port_depth' in scale_factors:
                scale_factors['fw_port_depth'] = max(scale_factors['fw_port_depth'], int(max_val))
            else:
                scale_factors['fw_port_depth'] = int(max_val)
        
        # Find max values for packet size if present
        if 'packet_size' in feature_cols:
            max_pkt_size = data['packet_size'].max()
            if 'packet_size' in scale_factors:
                scale_factors['packet_size'] = max(scale_factors['packet_size'], int(max_pkt_size))
            else:
                scale_factors['packet_size'] = int(max_pkt_size)
    
    # Ensure all features have a scale factor (default to 1)
    for feature in FEATURE_NAMES:
        if feature not in scale_factors:
            scale_factors[feature] = 1
    
    print("Global normalization parameters:", scale_factors)
    return scale_factors


def process_dataset(data, scale_factors, available_features):
    """
    Segments a dataset by switch, processes features, and returns ready-to-use segments.
    Ensures transitions only occur within the same switch's data.
    """
    segments = []
    
    # Sort by timestamp if available
    if 'timestamp' in data.columns:
        data = data.sort_values('timestamp')
    
    # Extract features, actions, rewards
    states = data[available_features].values
    actions = data['action'].values
    rewards = data['reward'].values
    
    # Create next_states by shifting
    if len(states) > 1:
        next_states = np.vstack([states[1:], states[-1]])
        
        # Mark only the last state as done
        dones = np.zeros(len(states), dtype=bool)
        dones[-1] = True
        
        segments.append((states, actions, rewards, next_states, dones))
        print(f"Added segment with {len(states)} transitions")
    
    return segments

def train_multi_scenario_dqn(csv_files, model_dir, epochs=100, batch_size=32):
    """
    Train a DQN agent using data from multiple network scenarios.
    No longer segments by switch_id.
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Calculate global normalization parameters
    scale_factors = get_global_normalization_parameters(csv_files)
    
    # Save scale factors for later use
    with open(os.path.join(model_dir, 'scale_factors.json'), 'w') as f:
        json.dump(scale_factors, f)
    
    # Get all available features across all datasets
    all_features = set()
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        all_features.update([col for col in FEATURE_NAMES if col in data.columns])
    
    # Process all datasets
    all_segments = []
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}")
        data = pd.read_csv(csv_file)
        
        # Check which features are in this dataset
        available_features = [col for col in all_features if col in data.columns]
        print(f"Available features: {available_features}")
        
        # For any missing features in this dataset, raise error
        for feature in FEATURE_NAMES:
            if feature not in data.columns and feature != 'switch_id':
                raise ValueError(f"Feature '{feature}' is missing in dataset {csv_file}")
                
        # Process dataset (no longer segmenting by switch_id)
        segments = process_dataset(data, scale_factors, available_features)
        
        # Add segments to overall collection
        all_segments.extend(segments)
        
        print(f"Added {len(segments)} segments from {csv_file}")
    
    # Determine state and action sizes
    state_size = len(all_features)
    
    # Find all unique actions across datasets
    all_actions = set()
    for segment in all_segments:
        all_actions.update(segment[1])  # actions are at index 1
    
    action_size = max(all_actions) + 1
    print(f"\nState size: {state_size}, Action size: {action_size}")
    
    # Combine all segments for training
    total_samples = sum(len(segment[0]) for segment in all_segments)
    print(f"Total training samples: {total_samples}")
    
    # Initialize the agent
    agent = DQNAgent(state_size, action_size, batch_size=batch_size)
    
    # Fill the agent's memory with the segmented datasets
    for segment in all_segments:
        states, actions, rewards, next_states, dones = segment
        for i in range(len(states)):
            agent.memorize(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    # Training loop
    losses = []
    for e in range(epochs):
        loss = agent.replay()
        losses.append(loss)
        
        if e % 10 == 0:
            print(f"Epoch {e}/{epochs}, Loss: {loss:.4f}")
        
        # Update target model periodically
        if e % 50 == 0:
            agent.update_target_model()
    
    # Save the trained model
    agent.save(os.path.join(model_dir, "dqn_model.h5"))
    
    # Also save the feature list and switch count for use during inference
    # with open(os.path.join(model_dir, 'feature_config.json'), 'w') as f:
    #     json.dump({
    #         'features': all_features,
    #         'switch_count': switch_count
    #     }, f)
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(model_dir, 'training_loss.png'))
    
    print(f"\nTraining completed. Model saved to {os.path.join(model_dir, 'dqn_model.h5')}")
    
    return agent, all_features


def main():
    parser = argparse.ArgumentParser(description='Train offline DQN agent with multiple scenarios')
    parser.add_argument('--model_dir', default='model', help='Directory to save the model')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--experiment_folder', default=None, help='Path to a specific experiment folder (overrides auto-discovery)')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)
    
    # Discover experiment datasets (auto or specific)
    csv_files = discover_experiment_datasets(args.experiment_folder)
    
    if not csv_files:
        print("Error: No experiment datasets found!")
        return
    
    print(f"Training on {len(csv_files)} datasets")
    
    # Train the agent on multiple datasets
    agent, all_features = train_multi_scenario_dqn(
        csv_files, 
        args.model_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size
    )
    
    # Load scale factors for evaluation
    with open(os.path.join(args.model_dir, 'scale_factors.json'), 'r') as f:
        scale_factors = json.load(f)


if __name__ == '__main__':
    main()