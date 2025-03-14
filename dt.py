import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rl_agent import DQNAgent, preprocess_dataset
from sklearn import tree
import pickle
import os
import tensorflow as tf
from tensorflow import keras
import sys
import json

def generate_dataset_from_dqn(dqn_model, states, num_samples=None):
    """
    Generate a dataset by having the DQN model make predictions on states.
    """
    if num_samples is None or num_samples > len(states):
        num_samples = len(states)
    
    # Randomly sample states if requested
    if num_samples < len(states):
        indices = np.random.choice(len(states), size=num_samples, replace=False)
        sampled_states = states[indices]
    else:
        sampled_states = states
    
    print(f"Generating DQN predictions for {num_samples} states")
    
    # Get predictions from DQN model
    dqn_actions = []
    for i, state in enumerate(sampled_states):
        # Report progress
        if i % 1000 == 0:
            print(f"Processing state {i}/{num_samples}")
        
        # Ensure state is converted to float32 for TensorFlow compatibility
        state_reshaped = state.reshape(1, -1).astype(np.float32)
        # Get action from DQN
        action = dqn_model.act(state_reshaped)
        dqn_actions.append(action)
    
    return sampled_states, np.array(dqn_actions)


def main():
    parser = argparse.ArgumentParser(description='Create Decision Tree from DQN')
    parser.add_argument('--csv', required=True, help='Path to the dataset CSV')
    parser.add_argument('--model', required=True, help='Path to the trained DQN model weights')
    parser.add_argument('--output_dir', default='dt_model', help='Directory to save the decision tree')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the decision tree')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to use (default: all)')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset
    print("Loading and preprocessing dataset...")
    states, actions, rewards, next_states, dones, feature_names = preprocess_dataset(args.csv)
    orig_actions = actions  # Rename for clarity
    
    # Extract feature columns from the CSV more generically
    data = pd.read_csv(args.csv)
    
    # Define columns to exclude (non-feature columns)
    exclude_columns = ['action', 'reward', 'done', 'timestamp']
    
    # Get all columns except those explicitly excluded
    feature_cols = [col for col in feature_names if col not in exclude_columns]
    
    # Print feature information for debugging
    print(f"Using {len(feature_cols)} features: {', '.join(feature_cols[:5])}{'...' if len(feature_cols) > 5 else ''}")
    
    # Get state and action dimensions
    state_size = states.shape[1]
    action_size = len(np.unique(orig_actions))
    
    print(f"State size: {state_size}, Action size: {action_size}")
    print(f"Feature columns: {len(feature_cols)}, State shape: {states.shape}")  # Debug print
    
    print(f"Dataset size: {len(states)} samples")
    
    # Initialize DQN agent
    print("Loading DQN model...")
    dqn_model = DQNAgent(state_size, action_size)
    dqn_model.load(args.model)
    
    # Generate dataset from DQN
    print("Building the DT dataset...")
    dqn_states, dqn_actions = generate_dataset_from_dqn(dqn_model, states, args.samples)
    
    # Save the dataset
    print("Saving dataset...")
    np.save(os.path.join(args.output_dir, "dt_states.npy"), dqn_states)
    np.save(os.path.join(args.output_dir, "dt_actions.npy"), dqn_actions)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        dqn_states, dqn_actions, test_size=0.2, random_state=42)
    
    # Train decision tree
    print("Training decision tree...")
    decision_tree = DecisionTreeClassifier(max_depth=args.max_depth)
    decision_tree.fit(X_train, y_train)
    
    # Evaluate decision tree
    y_pred = decision_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision tree accuracy (matching DQN): {accuracy * 100:.2f}%")
    
    # Compare with original dataset actions
    dqn_test_orig_actions = orig_actions[-len(y_test):]  # Using the same test set indices
    dt_vs_orig = accuracy_score(dqn_test_orig_actions, y_pred)
    print(f"Decision tree vs original actions match: {dt_vs_orig * 100:.2f}%")

    # Denormalize features
    try:
        # Load scale factors from the file saved during training
        with open('model/scale_factors.json', 'r') as f:
            scale_factors = json.load(f)
        
        print("\nDenormalizing features for interpretation...")

        # Get tree nodes and thresholds
        tree_structure = decision_tree.tree_
        feature_indices = tree_structure.feature
        thresholds = tree_structure.threshold
        
        # Only consider non-leaf nodes (where feature >= 0)
        valid_nodes = [i for i in range(len(feature_indices)) if feature_indices[i] >= 0]
        
        print("\nDenormalized Decision Thresholds (for top nodes):")
        for node_id in min(15, len(valid_nodes)):  
            feature_idx = feature_indices[node_id]
            feature_name = feature_cols[feature_idx]
            threshold = thresholds[node_id]
            
            if feature_name in scale_factors:
                denorm_threshold = threshold * scale_factors[feature_name]
                print(f"Node {node_id}: {feature_name} <= {denorm_threshold:.2f} (normalized: {threshold:.2f})")
            else:
                print(f"Node {node_id}: {feature_name} <= {threshold:.2f} (no scale factor found)")
    except Exception as e:
        print(f"Warning: Could not denormalize features: {e}")
    
    # Save the decision tree model
    dt_path = os.path.join(args.output_dir, "decision_tree.pkl")
    with open(dt_path, "wb") as file:
        pickle.dump(decision_tree, file)
    
    # Get feature importances
    importances = decision_tree.feature_importances_
    idx = np.argsort(importances)[::-1]
    top_features = [(feature_cols[i], importances[i]) for i in idx if importances[i] > 0.01]
    
    print("\nTop feature importances:")
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.title("Feature importances")
    plt.bar(range(len(idx)), importances[idx], align="center")
    plt.xticks(range(len(idx)), [feature_cols[i] for i in idx], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "feature_importance.png"))
    
    # Visualize the decision tree
    plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(decision_tree,
                       feature_names=feature_cols,
                       class_names=['normal', 'deflect'],
                       filled=True,
                       max_depth=3)  # Show only top levels for clarity
    
    plt.savefig(os.path.join(args.output_dir, "decision_tree.png"), dpi=300, bbox_inches='tight')
    
    print(f"Decision tree model saved to {dt_path}")
    print(f"Decision tree visualization saved to {os.path.join(args.output_dir, 'decision_tree.png')}")
    
    # Export the tree as a text representation (more interpretable for large trees)
    with open(os.path.join(args.output_dir, "decision_tree_rules.txt"), "w") as f:
        # Make sure feature_names length matches the number of features in the model
        if len(feature_cols) != states.shape[1]:
            print(f"WARNING: Feature columns count ({len(feature_cols)}) doesn't match state dimensions ({states.shape[1]})")
            # Use generic feature names if mismatch
            feature_names = [f"feature_{i}" for i in range(states.shape[1])]
        else:
            feature_names = feature_cols
            
        tree_rules = tree.export_text(decision_tree, 
                                     feature_names=feature_names,
                                     max_depth=5)
        f.write(tree_rules)
    
    print(f"Decision tree rules exported to {os.path.join(args.output_dir, 'decision_tree_rules.txt')}")

if __name__ == "__main__":
    main()