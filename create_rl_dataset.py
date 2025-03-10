#!/usr/bin/env python3

import pandas as pd
import argparse
import re
import csv
from collections import defaultdict
from datetime import datetime
import sys
import os

def parse_timestamp(ts_str):
    """Convert timestamp string to seconds."""
    ts_str = ts_str.strip("[]")
    dt = datetime.strptime(ts_str, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

def compute_reward(state):
    """Compute reward based on queue depths."""
    depths = [state[f'queue_depth_{p}'] for p in range(1, 8)]
    avg_depth = sum(depths) / len(depths) if depths else 0
    diff_depth = max(depths) - min(depths) if depths else 0
    # Reward as a negative weighted sum; lower queues yield higher reward
    reward = -sum(depths)
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
            print(f"Warning: {missing_mask.sum()} rows have missing reordering flags")
            print(f"Setting missing reordering_flag values to 0 (assuming in-order)")
            merged_df.loc[missing_mask, 'reordering_flag'] = 0
        
        # Ensure reordering_flag is an integer
        merged_df['reordering_flag'] = merged_df['reordering_flag'].astype(int)
        
        # Drop the temporary key column
        merged_df.drop(columns=['key'], inplace=True)
    else:
        print(f"Warning: Receiver log {receiver_csv} not found, using RL dataset without reordering info")
        merged_df = rl_df
        # Add reordering flag column with default value 0
        merged_df['reordering_flag'] = 0
    
    # Drop the flow_id, seq columns, and timestamp which are no longer needed
    merged_df.drop(columns=['flow_id', 'seq', 'timestamp'], inplace=True)
    
    # Incorporate reorder penalty into reward
    print("Adjusting rewards based on reordering flags...")
    def incorporate_reorder(row):
        # Apply a penalty to the reward if packet was reordered
        if row['reordering_flag'] == 1:
            return row['reward'] - 10  # Penalty for reordering
        else:
            return row['reward']
    
    merged_df['reward'] = merged_df.apply(incorporate_reorder, axis=1)
    
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
        window_size_us: Window size in microseconds (unused with simplified features)
        capacity_bps: Link capacity in bits per second (unused with simplified features)
    """
    print(f"Parsing switch log from {log_file}")
    
    # Ensure log file exists
    if not os.path.exists(log_file):
        print(f"Error: Switch log file {log_file} does not exist")
        return False
        
    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Regex patterns for parsing logs
    ingress_pattern = re.compile(r'Ingress: port=(\d+), size=(\d+), flow_id=(\d+), seq=(\d+), timestamp=(\d+)')
    deflection_pattern = re.compile(r'Deflection: original_port=(\d+), deflected_to=(\d+), random_number=(\d+)')
    normal_pattern = re.compile(r'Normal: port=(\d+)')
    # queue_pattern = re.compile(r'Queue: port=(\d+), deq_qdepth=(\d+)')
    queue_depths_pattern = re.compile(r'Queue depths: q0=(\d+) q1=(\d+) q2=(\d+) q3=(\d+) q4=(\d+) q5=(\d+) q6=(\d+) q7=(\d+)')
    
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
                event['port'] = int(ingress_match.group(1))
                event['size'] = int(ingress_match.group(2))
                event['flow_id'] = int(ingress_match.group(3))
                event['seq'] = int(ingress_match.group(4))
                event['timestamp'] = int(ingress_match.group(5))
            elif deflection_match := deflection_pattern.search(line):
                event['type'] = 'deflection'
                event['original_port'] = int(deflection_match.group(1))
                event['deflected_to'] = int(deflection_match.group(2))
                event['random_number'] = int(deflection_match.group(3))
            elif normal_match := normal_pattern.search(line):
                event['type'] = 'normal'
                event['port'] = int(normal_match.group(1))
            # elif queue_match := queue_pattern.search(line):
            #     event['type'] = 'queue'
            #     event['port'] = int(queue_match.group(1))
            #     event['deq_qdepth'] = int(queue_match.group(2))
            elif queue_depths_match := queue_depths_pattern.search(line):
                # TODO parametrize the number of egress ports
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
    
    # Track current state - only queue depths now
    current_queue_depth = {p: 0 for p in range(1, 8)}
    flow_id = 0
    seq = 0
    
    # Dataset creation
    dataset = []
    for event in events:
        # if event['type'] == 'queue':
        #     current_queue_depth[event['port']] = event['deq_qdepth']
        if event['type'] == 'queue_depths':
            for i, depth in enumerate(event['queue_depths']):
                current_queue_depth[i] = depth
        elif event['type'] == 'ingress':
            flow_id = event['flow_id']
            seq = event['seq']
        if event['type'] in ['deflection', 'normal']:
            # State: only queue depths for all ports
            state = {
                f'queue_depth_{p}': current_queue_depth[p] for p in range(1, 8)
            }
            state['flow_id'] = flow_id
            state['seq'] = seq
            
            deflected = 1 if event['type'] == 'deflection' else 0
            action = deflected
            
            # Initial reward (will be updated after merging with reordering info)
            reward = compute_reward(state)
            
            dataset.append({
                'timestamp': event['timestamp_str'],
                'state': state,
                'action': action,
                'reward': reward
            })
    
    # Write to CSV
    with open(output_csv, 'w', newline='') as csv_out:
        # Simplified headers - only queue depths
        headers = ['timestamp', 'action', 'reward'] + \
                  [f'queue_depth_{p}' for p in range(1, 8)] + \
                  ['flow_id', 'seq']
                  
        writer = csv.writer(csv_out)
        writer.writerow(headers)
        
        for entry in dataset:
            row = [entry['timestamp'], entry['action'], entry['reward']]
            row += [entry['state'][f'queue_depth_{p}'] for p in range(1, 8)]
            row += [entry['state']['flow_id']]
            row += [entry['state']['seq']]
            writer.writerow(row)
    
    print(f"Dataset with {len(dataset)} records written to {output_csv}")
    print("Note: Using simplified feature set with queue depths only for real-time RL processing")
    return True

def main():
    parser = argparse.ArgumentParser(description='Parse P4 switch logs and merge with receiver logs')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Parse logs subcommand
    parse_parser = subparsers.add_parser('parse', help='Parse P4 switch logs')
    parse_parser.add_argument('--log', default='log/p4s.s1.log', help='Path to P4 switch log file')
    parse_parser.add_argument('--output', required=True, help='Path to output RL dataset CSV')
    parse_parser.add_argument('--window', type=int, default=100000, 
                              help='Window size in microseconds (note: simplified feature set ignores this)')
    
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
    full_parser.add_argument('--intermediate', default='rl_dataset_temp.csv', help='Path for intermediate RL dataset')
    full_parser.add_argument('--window', type=int, default=100000, 
                             help='Window size in microseconds (note: simplified feature set ignores this)')
    
    args = parser.parse_args()
    
    if args.command == 'parse':
        parse_switch_log(args.log, args.output)
    elif args.command == 'merge':
        merge_by_flow_seq(args.rl, args.receiver, args.output)
    elif args.command == 'full':
        # Run the full pipeline
        os.makedirs(os.path.dirname(args.intermediate), exist_ok=True)
        if parse_switch_log(args.log, args.intermediate, args.window) is not False:
            merge_by_flow_seq(args.intermediate, args.receiver, args.output)
            print(f"Full pipeline complete! Intermediate file: {args.intermediate}, Final output: {args.output}")
        else:
            print(f"Pipeline stopped: Failed to process switch log {args.log}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()