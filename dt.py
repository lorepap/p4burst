import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os
import tensorflow as tf
from tensorflow import keras
import sys
import json
import glob

# Import necessary functions from rl_agent
from rl_agent import DQNAgent, FEATURE_NAMES, discover_experiment_datasets
from tqdm import tqdm


def generate_dataset_from_dqn(dqn_model, states, num_samples=None, batch_size=1024):
    """
    Generate a dataset by having the DQN model make predictions on states.
    Uses batch processing for much faster predictions.
    """
    if num_samples is None or num_samples > len(states):
        num_samples = len(states)
    
    # Randomly sample states if requested
    if num_samples < len(states):
        indices = np.random.choice(len(states), size=num_samples, replace=False)
        sampled_states = states[indices]
    else:
        sampled_states = states
    
    print(f"Generating DQN predictions for {num_samples} states using batch size {batch_size}")
    
    # Process in batches for much faster prediction
    dqn_actions = []
    total_batches = int(np.ceil(len(sampled_states) / batch_size))
    
    for i in tqdm(range(total_batches), desc="Processing batches"):
        # Report progress
        if i % 10 == 0:
            print(f"Processing batch {i+1}/{total_batches}")
        
        # Get the current batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sampled_states))
        batch = sampled_states[start_idx:end_idx].astype(np.float32)
        
        # Get predictions for the entire batch at once
        act_values = dqn_model.model.predict(batch, verbose=0)
        batch_actions = np.argmax(act_values, axis=1)
        dqn_actions.extend(batch_actions)
    
    return sampled_states, np.array(dqn_actions), indices if num_samples < len(states) else None


def preprocess_dataset(exp_folder=None):
    """
    Preprocess datasets from all experiments for DT knowledge distillation.
    Uses RAW FEATURE VALUES without normalization for better interpretability.
    """
    # Discover all experiment datasets
    dataset_paths = discover_experiment_datasets(exp_folder)
    if not dataset_paths:
        raise ValueError("No experiment datasets found!")
    
    print(f"Found {len(dataset_paths)} experiment datasets")
    
    # Get all available features across all datasets
    all_features = set()
    
    for path in dataset_paths:
        data = pd.read_csv(path)
        all_features.update([col for col in FEATURE_NAMES if col in data.columns])
    
    # Prepare combined dataset
    all_states = []
    all_actions = []
    
    for path in dataset_paths:
        print(f"Processing dataset: {path}")
        data = pd.read_csv(path)
        
        # Get available features for this dataset
        available_features = [col for col in FEATURE_NAMES if col in data.columns]
        
        # For any missing features, add zero columns
        for feature in FEATURE_NAMES:
            if feature not in data.columns:
                raise ValueError(f"Feature '{feature}' is missing in dataset {path}")
        
        # Use data without normalization - we're working with raw feature values
        data_processed = data.copy()
        
        # Extract states and actions
        all_states.append(data_processed[available_features].values)
        all_actions.append(data_processed['action'].values)
    
    # Combine all data
    combined_states = np.vstack(all_states)
    combined_actions = np.concatenate(all_actions)

    print(f"Combined dataset: {len(combined_states)} total samples")
    print(f"Using {len(available_features)} features in consistent order")
    print(f"Feature order: {available_features}")
    
    return combined_states, combined_actions, available_features


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataset with raw feature values (no normalization)
    print("Loading and preprocessing dataset with raw feature values...")
    states, actions, feature_cols = preprocess_dataset(args.exp_folder)
    orig_actions = actions 
    
    # Get state and action dimensions for DQN
    state_size = states.shape[1]
    action_size = len(np.unique(orig_actions))
    
    print(f"State size: {state_size}, Action size: {action_size}")
    
    # NOTE: When using the DQN for predictions, we need to normalize the input
    # since the DQN was trained on normalized data
    print("Loading scale factors for DQN input normalization...")
    with open('model/scale_factors.json', 'r') as f:
        scale_factors = json.load(f)
    
    # Create a normalized copy of the states specifically for DQN prediction
    states_for_dqn = states.copy()
    for i, col in enumerate(feature_cols):
        if col in scale_factors and scale_factors.get(col, 0) > 0:
            states_for_dqn[:, i] = states[:, i] / scale_factors[col]
    
    # Initialize DQN agent
    print("Loading DQN model...")
    dqn_model = DQNAgent(state_size, action_size)
    dqn_model.load(args.model)
    
    # Generate dataset from DQN using normalized inputs (as the DQN expects)
    print("Building the DT dataset using normalized features for DQN prediction...")
    dqn_states, dqn_actions, indices = generate_dataset_from_dqn(dqn_model, states_for_dqn, args.samples)
    
    # But we'll use the raw feature values for the decision tree
    # So we need to map the DQN's actions back to our raw feature values
    if args.samples is not None and args.samples < len(states):
        dqn_states = states[indices]  # Use raw feature values
    else:
        dqn_states = states  # Use all raw feature values
    
    # After denormalizing but before training the decision tree
    print("Converting features to appropriate types...")

    # List features that should be integers (not floating point)
    integer_features = []

    # Total queue depth is an integer
    if 'total_queue_depth' in feature_cols:
        integer_features.append('total_queue_depth')

    # Packet size is an integer
    if 'packet_size' in feature_cols:
        integer_features.append('packet_size')

    # Round and convert integer features
    for feature in integer_features:
        if feature in feature_cols:
            feature_idx = feature_cols.index(feature)
            print(f"  - Converting {feature} to integer type")
            # Round to nearest integer then convert to int32
            dqn_states[:, feature_idx] = np.round(dqn_states[:, feature_idx]).astype(np.int32)

    print(f"Converted {len(integer_features)} features to integer type")

    # Now split the dataset and train the decision tree
    X_train, X_test, y_train, y_test = train_test_split(
        dqn_states, dqn_actions, test_size=0.2, random_state=42)
    
    # Train decision tree
    print("Training decision tree...")
    decision_tree = DecisionTreeClassifier(max_depth=args.max_depth)
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)
    
    # Evaluate decision tree
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision tree accuracy (matching DQN): {accuracy * 100:.2f}%")
    
    # Compare with original dataset actions
    dqn_test_orig_actions = orig_actions[-len(y_test):]  # Using the same test set indices
    dt_vs_orig = accuracy_score(dqn_test_orig_actions, y_pred)
    print(f"Decision tree vs original actions match: {dt_vs_orig * 100:.2f}%")

    # Use actual sample count in filenames
    sample_count = len(dqn_states)
    tree_depth = decision_tree.get_depth()
    
    # Create a base filename with sample count and tree depth
    base_filename = f"dt_samples_{sample_count}_depth_{tree_depth}"
    
    # Save the dataset with sample count
    print("Saving dataset...")
    np.save(os.path.join(args.output_dir, f"{base_filename}_states.npy"), dqn_states)
    np.save(os.path.join(args.output_dir, f"{base_filename}_actions.npy"), dqn_actions)

    # Save the decision tree model with sample count and depth
    dt_path = os.path.join(args.output_dir, f"{base_filename}.pkl")
    with open(dt_path, "wb") as file:
        pickle.dump(decision_tree, file)
    
    # Save the feature names with same naming convention
    feature_names_path = os.path.join(args.output_dir, f"{base_filename}_feature_names.pkl")
    with open(feature_names_path, "wb") as file:
        pickle.dump(feature_cols, file)
    print(f"Feature names saved to {feature_names_path}")

    # Save as JSON too for easier inspection
    feature_names_json_path = os.path.join(args.output_dir, f"{base_filename}_feature_names.json")
    with open(feature_names_json_path, "w") as file:
        json.dump(feature_cols, file, indent=2)
    print(f"Feature names also saved in JSON format to {feature_names_json_path}")

    # Also save feature dimensions information for debugging
    features_info = {
        "model_features": decision_tree.n_features_in_,
        "feature_names_count": len(feature_cols),
        "samples_used": sample_count,
        "tree_depth": tree_depth,
        "accuracy": float(accuracy)
    }
    
    with open(os.path.join(args.output_dir, f"{base_filename}_info.json"), "w") as file:
        json.dump(features_info, file, indent=2)
    
    # Get feature importances
    importances = decision_tree.feature_importances_
    idx = np.argsort(importances)[::-1]
    top_features = [(feature_cols[i], importances[i]) for i in idx if importances[i] > 0.01]
    
    print("\nTop feature importances:")
    for feature, importance in top_features:
        print(f"{feature}: {importance:.4f}")
    
    # Plot feature importances with sample count and depth in filename
    plt.figure(figsize=(12, 6))
    plt.title(f"Feature importances (samples: {sample_count}, depth: {tree_depth})")
    plt.bar(range(len(idx)), importances[idx], align="center")
    plt.xticks(range(len(idx)), [feature_cols[i] for i in idx], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{base_filename}_feature_importance.png"))
    
    # Visualize the decision tree with sample count and depth in filename
    plt.figure(figsize=(25, 20))
    import sklearn.tree as tree
    _ = tree.plot_tree(decision_tree,
                       feature_names=feature_cols,
                       class_names=['action_0', 'action_1'],
                       filled=True,
                       max_depth=None)  # Show only top levels for clarity
    
    plt.savefig(os.path.join(args.output_dir, f"{base_filename}_tree.png"), dpi=300, bbox_inches='tight')
    
    print(f"Decision tree model saved to {dt_path}")
    print(f"Decision tree visualization saved to {os.path.join(args.output_dir, f'{base_filename}_tree.png')}")
    
    # Export the tree as a text representation with sample count and depth in filename
    with open(os.path.join(args.output_dir, f"{base_filename}_rules.txt"), "w") as f:
        # Make sure feature_names length matches the number of features in the model
        if len(feature_cols) != dqn_states.shape[1]:
            print(f"WARNING: Feature columns count ({len(feature_cols)}) doesn't match DT feature dimensions ({dqn_states.shape[1]})")
            # Use generic feature names if mismatch, but with the CORRECT feature count
            feature_names = [f"feature_{i}" for i in range(dqn_states.shape[1])]
        else:
            feature_names = feature_cols
            
        # Verify again before exporting
        print(f"Decision tree has {decision_tree.n_features_in_} features, providing {len(feature_names)} feature names")
        
        # Make absolutely sure the feature names count matches the model's feature count
        if len(feature_names) != decision_tree.n_features_in_:
            print(f"CRITICAL: Feature names count still doesn't match model! Generating generic names.")
            feature_names = [f"feature_{i}" for i in range(decision_tree.n_features_in_)]
        
        tree_rules = tree.export_text(decision_tree, 
                                     feature_names=feature_names,
                                     max_depth=5)
        f.write(tree_rules)
    
    print(f"Decision tree rules exported to {os.path.join(args.output_dir, f'{base_filename}_rules.txt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create Decision Tree from DQN')
    # Make --csv optional (None value indicates automatic dataset selection)
    parser.add_argument('--exp_folder', default=None, help='Path to the experiment folder (optional, will auto-discover if not provided)')
    parser.add_argument('--model', default='model/dqn_model.h5', help='Path to the trained DQN model weights')
    parser.add_argument('--output_dir', default='dt_model', help='Directory to save the decision tree')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the decision tree')
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to use (default: all)')
    parser.add_argument('--denormalize', action='store_true', help='Denormalize features before DT training for better interpretability')
    args = parser.parse_args()
    main(args)