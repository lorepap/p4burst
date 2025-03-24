import os
import pandas as pd
import re
import csv
from datetime import datetime
import sys
sys.path.append('/home/ubuntu/p4burst')
from topology import LeafSpineTopology
import traceback

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
    pkt_cnt = 0
    with open(log_file, 'r') as log:
        for line in log:
            parts = line.split()
            if not parts:
                continue
                
            timestamp_str = parts[0]
            event = {'timestamp_str': timestamp_str}
            
            # Parse event type and details
            if ingress_match := ingress_pattern.search(line):
                pkt_cnt += 1
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

    print(f"Found {pkt_cnt} packets in the log")
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

def collect_switch_logs(topology: LeafSpineTopology, exp_dir):
    """Collect and process switch logs"""
    # Get log paths for all switches
    sw_logs = [f"log/p4s.{sw}.log" for sw in topology.net.switches()]
    
    # Process each switch log
    switch_datasets = []
    for i, switch_log in enumerate(sw_logs):
        if os.path.exists(switch_log):
            print(f"Processing switch log: {switch_log}")
            intermediate_dataset = f"{exp_dir}/s{i+1}_rl_dataset.csv"
            
            if parse_switch_log(switch_log, intermediate_dataset):
                switch_datasets.append(intermediate_dataset)
            else:
                print(f"Failed to process switch log: {switch_log}")
        else:
            print(f"Warning: Switch log {switch_log} not found")
    
    return switch_datasets

def merge_by_flow_seq(switch_csv, receiver_csv, output_csv):
    """
    Merge RL dataset with receiver logs to incorporate reordering information and packet type.
    
    Args:
        switch_csv: Path to the RL dataset CSV from parse_switch_log()
        receiver_csv: Path to the receiver log CSV with reordering flags and packet types
        output_csv: Path to write the final merged dataset
    """
    print(f"Merging RL dataset {switch_csv} with receiver log {receiver_csv}")
    
    # Read RL dataset
    try:
        switch_df = pd.read_csv(switch_csv)
    except:
        print(f"Error: Unable to read RL dataset {switch_csv}")
        traceback.print_exc()
        return False
    
    # Read receiver logs
    if os.path.exists(receiver_csv):
        receiver_df = pd.read_csv(receiver_csv)
        
        # Convert the reordering_flag column to integer explicitly
        receiver_df['reordering_flag'] = receiver_df['reordering_flag'].astype(int)
        
        # Check if packet_type exists in receiver logs, if not create with default value
        if 'packet_type' not in receiver_df.columns:
            raise ValueError("packet_type column not found in receiver logs")
        # Packet type as an integer
        receiver_df['packet_type'] = receiver_df['packet_type'].astype(int)
        receiver_df['packet_size'] = receiver_df['packet_size'].astype(int)
        
        # Create a combined key for merging
        receiver_df['key'] = receiver_df['flow_id'].astype(str) + '_' + receiver_df['seq'].astype(str) + '_' + receiver_df['packet_size'].astype(str)
        switch_df['key'] = switch_df['flow_id'].astype(str) + '_' + switch_df['seq'].astype(str) + '_' + switch_df['packet_size'].astype(str)
        
        # Merge the datasets
        merged_df = pd.merge(switch_df, 
                             receiver_df[['key', 'reordering_flag', 'packet_type']], 
                             on='key', how='left')
        
        # Identify rows where reordering_flag or packet_type is missing
        missing_mask = merged_df['reordering_flag'].isna() | merged_df['packet_type'].isna()
        missing_count = missing_mask.sum()
        
        if missing_count > 0:
            print(f"Found {missing_count} packets in switch log that aren't in receiver log")
            
            # Keep only packets that have corresponding entries in the receiver log
            merged_df = merged_df[~missing_mask].copy()
            print(f"Removed {missing_count} packets from dataset that weren't received")
            
            if merged_df.empty:
                print("Error: No packets remain after filtering")
                return False
        
        # Drop the temporary key column
        merged_df.drop(columns=['key'], inplace=True)
    else:
        print(f"Error: Receiver log {receiver_csv} not found")
        return False
    
    # Now compute rewards after all features are available
    print("Computing rewards based on queue depths and reordering flags...")
    merged_df['reward'] = merged_df.apply(compute_reward, axis=1)
    
    # Rearrange columns to put reward after action and keep packet_type
    cols = merged_df.columns.tolist()
    cols.remove('reward')
    cols.insert(cols.index('action') + 1, 'reward')
    
    # Keep packet_type but remove flow_id and seq which are no longer needed
    cols = [col for col in cols if col not in ['seq', 'flow_id']]
    merged_df = merged_df[cols]

    # For clarity - only include columns that exist
    final_cols = []
    if 'timestamp' in merged_df.columns:
        final_cols.append('timestamp')
    final_cols.extend(['action', 'reward'])
    final_cols.extend([f'queue_depth_{p+1}' for p in range(8) if f'queue_depth_{p+1}' in merged_df.columns])
    final_cols.extend([f'queue_occ_{p+1}' for p in range(8) if f'queue_occ_{p+1}' in merged_df.columns])
    final_cols.extend(['packet_size', 'switch_id', 'reordering_flag', 'packet_type'])
    
    # Only include columns that actually exist
    final_cols = [col for col in final_cols if col in merged_df.columns]
    merged_df = merged_df[final_cols]
    
    # Write the final CSV
    print(f"Writing merged dataset to {output_csv}")
    merged_df.to_csv(output_csv, index=False)
    print(f"Merge complete: {len(merged_df)} total rows in final dataset")
    return True

def combine_datasets(switch_datasets, host_logs, exp_dir):
    """Combine switch and receiver csv files, then merge them into the final RL dataset"""
    
    if not switch_datasets:
        print("Warning: No valid switch logs found")
        return False

    # File names
    combined_switch_csv = f"{exp_dir}/combined_switch_dataset.csv"
    
    # Look for client response logs too
    # sender_logs = [f for f in os.listdir(exp_dir) if f.startswith('sender_') and f.endswith('_dataset.csv')]
    # if sender_logs:
    #     print(f"Found {len(sender_logs)} sender logs")
    #     all_logs = receiver_logs + [f"{exp_dir}/{log}" for log in sender_logs]
    # else:
    #     all_logs = receiver_logs
        
    final_receiver_csv = f"{exp_dir}/combined_receiver_log.csv"
    
    # Initialize dataframes
    receiver_df = None
    combined_df = None
    
    # Read and combine switch datasets
    for dataset in switch_datasets:
        if os.path.exists(dataset):
            print(f"Reading switch dataset: {dataset}")
            df = pd.read_csv(dataset)
            if combined_df is None:
                combined_df = df
            else:
                combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    if combined_df is None or combined_df.empty:
        print("Warning: No valid switch datasets found")
        return False

    # Save combined switch dataset
    combined_df.to_csv(combined_switch_csv, index=False)
    print(f"Combined {len(switch_datasets)} switch datasets into {combined_switch_csv}")
    
    # Read and combine receiver logs (including client response logs)
    for log_file in host_logs:
        if os.path.exists(log_file):
            print(f"Reading log file: {log_file}")
            df = pd.read_csv(log_file)
            if receiver_df is None:
                receiver_df = df
            else:
                receiver_df = pd.concat([receiver_df, df], ignore_index=True)
    
    if receiver_df is None or receiver_df.empty:
        print("Warning: No valid receiver log data found")
        return False
    
    # Save combined receiver log
    receiver_df.to_csv(final_receiver_csv, index=False)
    print(f"Combined {len(host_logs)} log files into {final_receiver_csv}")

    # Merge the combined switch dataset with the receiver log
    if not merge_by_flow_seq(combined_switch_csv, final_receiver_csv, f"{exp_dir}/final_dataset.csv"):
        print("Error: Failed to merge datasets")
        return False
    
    print(f"Final dataset created at {exp_dir}/final_dataset.csv")
    return True


if __name__ == "__main__":
    print("Generating final dataset...")
    exp_dir = 'tmp/20250317_212554'
    os.makedirs(exp_dir, exist_ok=True)
    topology = LeafSpineTopology(
            num_hosts=2, 
            num_leaf=2, 
            num_spine=2, 
            bw=10, 
            latency=0.01,
            p4_program="p4src/sd/sd.p4"
    )
    topology.generate_topology()
    switch_datasets = collect_switch_logs(topology, exp_dir)
    receiver_logs = [f'{exp_dir}/receiver_log.csv']
    combined_switch_dataset, combined_receiver_dataset = combine_datasets(switch_datasets, receiver_logs, exp_dir)
    final_dataset_file = f"{exp_dir}/final_dataset.csv"
    if merge_by_flow_seq(combined_switch_dataset, combined_receiver_dataset, final_dataset_file):
        print(f"Created final dataset: {final_dataset_file}")
    print.info(f"Final dataset created at {exp_dir}/combined_dataset.csv")
    print("RL pipeline execution completed.")