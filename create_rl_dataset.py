#!/usr/bin/env python3

import pandas as pd
import argparse
import re
import csv
from collections import defaultdict
from datetime import datetime
import sys
import os

# Feature names for the RL dataset
FEATURE_NAMES = (
    [f'queue_depth_{p+1}' for p in range(8)] + 
    [f'queue_occ_{p+1}' for p in range(8)] + 
    ['packet_size', 'switch_id']
)

# All column names including metadata and targets
ALL_COLUMN_NAMES = (
    ['timestamp', 'action', 'reward'] + 
    FEATURE_NAMES + 
    ['flow_id', 'seq']
)

def parse_timestamp(ts_str):
    """Convert timestamp string to seconds."""
    ts_str = ts_str.strip("[]")
    dt = datetime.strptime(ts_str, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

def compute_reward(row):
    """
    Compute reward based on queue depths and reordering.
    Lower queue depths are better, reordering incurs penalties.
    """
    # Extract queue depths (skip port 0 which starts with index 1)
    queue_depths = [row[f'queue_depth_{p+1}'] for p in range(8)]
    
    # Basic reward: negative sum of queue depths (lower is better)
    reward = -sum(queue_depths)
    
    # Apply penalty for reordering
    if row['reordering_flag'] == 1:
        reward -= 10  # Significant penalty for reordering
        
    return reward

def merge_by_flow_seq(rl_csv, receiver_csv, output_csv):
    """
    Merge RL dataset with receiver logs to incorporate reordering information.
    
    Args:
        rl_csv: Path to the RL dataset CSV from parse_switch_log()
        receiver_csv: Path to the receiver log CSV with reordering flags
        output_csv: Path to write the final merged dataset
    """
    print(f"Merging RL dataset {rl_csv} with receiver log {receiver_csv}")
    
    # Read RL dataset
    rl_df = pd.read_csv(rl_csv)
    
    # Read receiver logs
    if os.path.exists(receiver_csv):
        receiver_df = pd.read_csv(receiver_csv)
        
        # Convert the reordering_flag column to integer explicitly
        receiver_df['reordering_flag'] = receiver_df['reordering_flag'].astype(int)
        
        # Create a combined key for merging
        receiver_df['key'] = receiver_df['flow_id'].astype(str) + '_' + receiver_df['seq'].astype(str)
        
        # Create similar key in the RL dataset
        rl_df['key'] = rl_df['flow_id'].astype(str) + '_' + rl_df['seq'].astype(str)
        
        # Merge the datasets
        merged_df = pd.merge(rl_df, receiver_df[['key', 'reordering_flag']], on='key', how='left')
        
        # Identify rows where reordering_flag is missing
        missing_mask = merged_df['reordering_flag'].isna()
        if missing_mask.sum() > 0:
            #raise ValueError(f"Error: {missing_mask.sum()} rows have missing reordering flags")
            print(f"Warning: {missing_mask.sum()} rows have missing reordering flags, prolly due to missing data in the switch log.")
            print("Skipping merged csv")
            #merged_df.loc[missing_mask, 'reordering_flag'] = 0
            return False
        
        # Ensure reordering_flag is an integer
        merged_df['reordering_flag'] = merged_df['reordering_flag'].astype(int)
        
        # Drop the temporary key column
        merged_df.drop(columns=['key'], inplace=True)
    else:
        print(f"Warning: Receiver log {receiver_csv} not found, using RL dataset without reordering info")
        merged_df = rl_df
        # Add reordering flag column with default value 0
        merged_df['reordering_flag'] = 0
    
    # Drop unnecessary columns
    if 'timestamp' in merged_df.columns:
        merged_df.drop(columns=['timestamp'], inplace=True)
    
    # Now compute rewards after all features are available
    print("Computing rewards based on queue depths and reordering flags...")
    merged_df['reward'] = merged_df.apply(compute_reward, axis=1)
    
    # Rearrange columns to put reward after action
    cols = merged_df.columns.tolist()
    cols.remove('reward')
    cols.insert(cols.index('action') + 1, 'reward')
    merged_df = merged_df[cols]
    merged_df.drop(columns=['seq', 'flow_id'], inplace=True)
    
    # Write the final CSV
    print(f"Writing merged dataset to {output_csv}")
    merged_df.to_csv(output_csv, index=False)
    print(f"Merge complete: {len(merged_df)} total rows in final dataset")
    return True

def parse_switch_log(log_file, output_csv):
    """
    Parse P4 switch logs to extract flow data and create an RL dataset.
    
    Args:
        log_file: Path to the P4 switch log file
        output_csv: Path to write the output RL dataset CSV
    """
    print(f"Parsing switch log from {log_file}")
    
    # Ensure log file exists
    if not os.path.exists(log_file):
        print(f"Error: Switch log file {log_file} does not exist")
        return False
        
    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Regex patterns for parsing logs
    # Updated ingress pattern to capture source IP, destination IP, and packet size
    ingress_pattern = re.compile(r'Ingress: switch_id=(\d+), port=(\d+), size=(\d+), flow_id=(\d+), seq=(\d+), timestamp=(\d+)')
    deflection_pattern = re.compile(r'Deflection: original_port=(\d+), deflected_to=(\d+), random_number=(\d+)')
    normal_pattern = re.compile(r'Normal: port=(\d+)')
    queue_depths_pattern = re.compile(r'Queue depths: q0=(\d+) q1=(\d+) q2=(\d+) q3=(\d+) q4=(\d+) q5=(\d+) q6=(\d+) q7=(\d+)')
    queue_occ_pattern = re.compile(r'Queue occupancy: q0=(\d+) q1=(\d+) q2=(\d+) q3=(\d+) q4=(\d+) q5=(\d+) q6=(\d+) q7=(\d+)')
    
    # Data storage for events
    events = []
    with open(log_file, 'r') as log:
        for line in log:
            parts = line.split()
            if not parts:
                continue
                
            timestamp_str = parts[0]
            event = {'timestamp_str': timestamp_str}
            
            # Parse event type and details
            if ingress_match := ingress_pattern.search(line):
                event['type'] = 'ingress'
                event['switch_id'] = int(ingress_match.group(1))
                event['port'] = int(ingress_match.group(2))
                event['size'] = int(ingress_match.group(3))  # Packet size
                event['flow_id'] = int(ingress_match.group(4))
                event['seq'] = int(ingress_match.group(5))
                event['timestamp'] = int(ingress_match.group(6))
            elif deflection_match := deflection_pattern.search(line):
                event['type'] = 'deflection'
                event['original_port'] = int(deflection_match.group(1))
                event['deflected_to'] = int(deflection_match.group(2))
                event['random_number'] = int(deflection_match.group(3))
            elif normal_match := normal_pattern.search(line):
                event['type'] = 'normal'
                event['port'] = int(normal_match.group(1))
            elif queue_depths_match := queue_depths_pattern.search(line):
                event['type'] = 'queue_depths'
                event['queue_depths'] = [
                    int(queue_depths_match.group(1)),  # q0
                    int(queue_depths_match.group(2)),  # q1
                    int(queue_depths_match.group(3)),  # q2
                    int(queue_depths_match.group(4)),  # q3
                    int(queue_depths_match.group(5)),  # q4
                    int(queue_depths_match.group(6)),  # q5
                    int(queue_depths_match.group(7)),  # q6
                    int(queue_depths_match.group(8))   # q7
                ]
            elif queue_occ_match := queue_occ_pattern.search(line):
                event['type'] = 'queue_occ'
                event['queue_occ'] = [
                    int(queue_occ_match.group(1)),  # q0
                    int(queue_occ_match.group(2)),  # q1
                    int(queue_occ_match.group(3)),  # q2
                    int(queue_occ_match.group(4)),  # q3
                    int(queue_occ_match.group(5)),  # q4
                    int(queue_occ_match.group(6)),  # q5
                    int(queue_occ_match.group(7)),  # q6
                    int(queue_occ_match.group(8))   # q7
                ]
                
            if 'type' in event:
                events.append(event)
    
    # Sort events by timestamp
    events.sort(key=lambda x: parse_timestamp(x.get('timestamp_str', "[00:00:00.000]")))
    
    # Find the first timestamp for normalization
    t0 = None
    for event in events:
        if event['type'] == 'ingress':
            t0 = event['timestamp']
            break
    if t0 is None:
        print("Error: No ingress events found in the log.")
        return False
    
    # Track current state
    current_queue_depth = {p: 0 for p in range(8)}
    current_queue_occ = {p: 0 for p in range(8)}
    
    # Dataset creation
    dataset = []
    for event in events:
        if event['type'] == 'queue_depths':
            for i, depth in enumerate(event['queue_depths']):
                current_queue_depth[i] = depth
        if event['type'] == 'queue_occ':
            for i, occ in enumerate(event['queue_occ']):
                current_queue_occ[i] = occ
        elif event['type'] == 'ingress':
            switch_id = event['switch_id']
            flow_id = event['flow_id']
            seq = event['seq']
            packet_size = event['size']  # Keep packet size
        if event['type'] in ['deflection', 'normal']:
            # First create a record with just the state features and action
            # Reward will be computed after merging with receiver logs
            action = 1 if event['type'] == 'deflection' else 0
            
            # Create data entry with all features
            entry = {
                'timestamp': event['timestamp_str'],
                'action': action,
                'reward': 0,  # Placeholder, will be computed later
                'flow_id': flow_id,
                'seq': seq,
                'packet_size': packet_size,
                'switch_id': switch_id
            }
            
            # Add queue depths for all ports
            for p in range(8):
                entry[f'queue_depth_{p+1}'] = current_queue_depth[p]
                entry[f'queue_occ_{p+1}'] = current_queue_occ[p]
            
            dataset.append(entry)
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csv_out:
        headers = ['timestamp', 'action', 'reward'] + \
                  [f'queue_depth_{p+1}' for p in range(8)] + \
                  [f'queue_occ_{p+1}' for p in range(8)] + \
                  ['packet_size', 'switch_id', 'flow_id', 'seq']
                  
        writer = csv.DictWriter(csv_out, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dataset)
    
    print(f"Initial dataset with {len(dataset)} records written to {output_csv}")
    print("Note: Rewards are placeholders, will be computed after merging with receiver logs")
    return True

def main():
    parser = argparse.ArgumentParser(description='Parse P4 switch logs and merge with receiver logs')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parse logs subcommand
    parse_parser = subparsers.add_parser('parse', help='Parse P4 switch logs')
    parse_parser.add_argument('--log', default='log/p4s.s1.log', help='Path to P4 switch log file')
    parse_parser.add_argument('--output', required=True, help='Path to output RL dataset CSV')
    parse_parser.add_argument('--window', type=int, default=100000, 
                              help='Window size in microseconds (not used in this version)')
    
    # Merge logs subcommand
    merge_parser = subparsers.add_parser('merge', help='Merge RL dataset with receiver logs')
    merge_parser.add_argument('--rl', required=True, help='Path to RL dataset CSV')
    merge_parser.add_argument('--receiver', required=True, help='Path to receiver log CSV')
    merge_parser.add_argument('--output', required=True, help='Path for merged output CSV')
    
    # Full pipeline subcommand (parse + merge)
    full_parser = subparsers.add_parser('full', help='Run full pipeline: parse logs then merge with receiver logs')
    full_parser.add_argument('--log', required=True, help='Path to P4 switch log file')
    full_parser.add_argument('--receiver', required=True, help='Path to receiver log CSV')
    full_parser.add_argument('--output', required=True, help='Path for final output CSV')
    full_parser.add_argument('--intermediate', default='tmp/rl_dataset_temp.csv', help='Path for intermediate RL dataset')
    
    args = parser.parse_args()
    
    if args.command == 'parse':
        parse_switch_log(args.log, args.output)
    elif args.command == 'merge':
        merge_by_flow_seq(args.rl, args.receiver, args.output)
    elif args.command == 'full':
        # Run the full pipeline
        os.makedirs(os.path.dirname(args.intermediate), exist_ok=True)
        if parse_switch_log(args.log, args.intermediate) is not False:
            merge_by_flow_seq(args.intermediate, args.receiver, args.output)
            print(f"Full pipeline complete! Intermediate file: {args.intermediate}, Final output: {args.output}")
        else:
            print(f"Pipeline stopped: Failed to process switch log {args.log}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()