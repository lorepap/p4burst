import os
import pandas as pd
import re
import csv
from datetime import datetime


def parse_timestamp(ts_str):
    """Convert timestamp string to seconds."""
    ts_str = ts_str.strip("[]")
    dt = datetime.strptime(ts_str, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

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

def collect_switch_logs(topology, exp_dir):
    """Collect and process switch logs"""
    # Get log paths for all switches
    sw_logs = [f"log/p4s.{sw.name}.log" for sw in topology.net.net.switches]
    
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

def combine_datasets(switch_datasets, receiver_logs, exp_dir):
    """Combine all datasets into a final RL dataset"""
    if not switch_datasets:
        print("Warning: No valid switch logs found")
        return None
        
    # Combine switch datasets
    combined_switch_dataset = f"{exp_dir}/combined_switch_dataset.csv"
    combined_df = None
    
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
        return None
        
    # Save combined switch dataset
    combined_df.to_csv(combined_switch_dataset, index=False)
    print(f"Combined {len(switch_datasets)} switch datasets into {combined_switch_dataset}")
    
    # Combine receiver logs
    combined_receiver_log = f"{exp_dir}/combined_receiver_log.csv"
    receiver_df = None
    
    for log_file in receiver_logs:
        if os.path.exists(log_file):
            print(f"Reading receiver log: {log_file}")
            df = pd.read_csv(log_file)
            if receiver_df is None:
                receiver_df = df
            else:
                receiver_df = pd.concat([receiver_df, df], ignore_index=True)
    
    if receiver_df is None or receiver_df.empty:
        print("Warning: No valid receiver log data found")
        return None


if __name__ == "__main__":
    
    print("RL pipeline execution completed.")