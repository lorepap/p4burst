"""
TODO:
- update reward with fw port depth
"""

import os
import pandas as pd
import re
import csv
from datetime import datetime
import sys
sys.path.append('/home/ubuntu/p4burst')
from topology import LeafSpineTopology
import traceback
import pyshark
import statistics
import io
import subprocess

# Feature names for the RL dataset
FEATURE_NAMES = (
    ['total_queue_depth', 'packet_size']  # Simplified feature set
)

# All column names including metadata and targets
ALL_COLUMN_NAMES = (
    ['timestamp', 'action', 'reward'] + 
    FEATURE_NAMES + 
    ['flow_id', 'seq']
)

def int_to_ip(ip_int):
    """
    Convert an integer IP address to the dotted-decimal string format.
    
    Args:
        ip_int: Integer representation of an IP address
        
    Returns:
        String representation in X.X.X.X format
    """
    # Convert string to int if needed
    if isinstance(ip_int, str):
        ip_int = int(ip_int)
        
    # Extract octets - network byte order (big-endian)
    octet1 = (ip_int >> 24) & 0xFF
    octet2 = (ip_int >> 16) & 0xFF
    octet3 = (ip_int >> 8) & 0xFF
    octet4 = ip_int & 0xFF
    
    # Format as dotted decimal in correct order
    return f"{octet1}.{octet2}.{octet3}.{octet4}"

def parse_timestamp(ts_str):
    """Convert timestamp string to seconds."""
    ts_str = ts_str.strip("[]")
    dt = datetime.strptime(ts_str, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

def compute_reward(row):
    """
    Reward penalizes:
    - High forward port depth when NOT deflecting (missed opportunity).
    - High total queue depth (general congestion).
    - Packet reordering strongly discouraged.
    - High RTT values (latency penalty).
    """
    
    fw_port_depth = row['fw_port_depth']
    total_queue_depth = row['total_queue_depth']
    out_of_order_flag = row['out_of_order_flag']
    action = row['action']  # 1=deflect, 0=forward
    rtt = row['rtt']  # RTT in milliseconds
    
    reward = 0
    
    # 1. Penalize general congestion (total queue depth)
    reward -= total_queue_depth * 0.5  # general penalty factor
    
    # 2. Burst-aware penalty: forward port depth
    # Strong penalty if forward port depth is high and action is NOT deflecting
    if action == 0:
        reward -= fw_port_depth * 2.0  # strong penalty (missed deflection during burst)
    else:
        # small penalty proportional to forward port depth (encourage deflection only when needed)
        reward -= fw_port_depth * 0.5
    
    # 3. Heavy penalty for out-of-order packets
    if out_of_order_flag == 1:
        reward -= 10  # strong penalty to discourage unnecessary deflection
    
    # 4. RTT penalty - higher RTT means worse reward
    rtt_penalty_factor = 2  # Adjust this factor to control RTT penalty strength
    reward -= rtt * rtt_penalty_factor
    
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
    ingress_pattern = re.compile(r'Ingress: port=(\d+), size=(\d+), timestamp=(\d+)')
    deflection_pattern = re.compile(r'Deflection: original_port=(\d+), deflected_to=(\d+), random_number=(\d+), fw_port_depth=(\d+)')
    normal_pattern = re.compile(r'Normal: port=(\d+)')
    fw_port_depth_pattern = re.compile(r'Forward port depth: port=(\d+), depth=(\d+)')
    queue_depths_pattern = re.compile(r'Queue depths: q0=(\d+) q1=(\d+) q2=(\d+) q3=(\d+) q4=(\d+) q5=(\d+) q6=(\d+) q7=(\d+)')
    # Add TCP pattern
    tcp_pattern = re.compile(r'TCP: src_ip=(\d+), dst_ip=(\d+), src_port=(\d+), dst_port=(\d+), seq=(\d+)')
    
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
                event['port'] = int(ingress_match.group(1))
                event['size'] = int(ingress_match.group(2))  # Packet size
                event['timestamp'] = int(ingress_match.group(3))
            elif tcp_match := tcp_pattern.search(line):
                event['type'] = 'tcp'
                src_ip_int = tcp_match.group(1)
                dst_ip_int = tcp_match.group(2)
                event['src_ip'] = int_to_ip(src_ip_int)
                event['dst_ip'] = int_to_ip(dst_ip_int)
                event['src_port'] = int(tcp_match.group(3))
                event['dst_port'] = int(tcp_match.group(4))
                event['tcp_seq'] = int(tcp_match.group(5))
            elif deflection_match := deflection_pattern.search(line):
                event['type'] = 'deflection'
                event['original_port'] = int(deflection_match.group(1))
                event['deflected_to'] = int(deflection_match.group(2))
                event['random_number'] = int(deflection_match.group(3))
                event['fw_port_depth'] = int(deflection_match.group(4))
            elif normal_match := normal_pattern.search(line):
                event['type'] = 'normal'
                event['port'] = int(normal_match.group(1))
            elif fw_port_depth_match := fw_port_depth_pattern.search(line):
                event['type'] = 'fw_port_depth'
                event['port'] = int(fw_port_depth_match.group(1))
                event['depth'] = int(fw_port_depth_match.group(2))
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
    total_queue_depth = 0
    fw_port_depth = 0  # Initialize fw_port_depth
    fw_port_depths = {}  # Dictionary to store fw_port_depth by port number
    
    # TCP information
    current_tcp = {
        'src_ip': None,
        'dst_ip': None,
        'src_port': None,
        'dst_port': None,
        'tcp_seq': None
    }
    
    # Dataset creation
    dataset = []
    for event in events:
        if event['type'] == 'queue_depths':
            for i, depth in enumerate(event['queue_depths']):
                current_queue_depth[i] = depth
            total_queue_depth = sum(current_queue_depth.values())
        elif event['type'] == 'tcp':
            # Update current TCP information
            current_tcp['src_ip'] = event['src_ip']
            current_tcp['dst_ip'] = event['dst_ip']
            current_tcp['src_port'] = event['src_port']
            current_tcp['dst_port'] = event['dst_port']
            current_tcp['tcp_seq'] = event['tcp_seq']
        elif event['type'] == 'ingress':
            packet_size = event['size']
        elif event['type'] == 'fw_port_depth':
            # Store the fw_port_depth for the port
            fw_port_depths[event['port']] = event['depth']
        
        if event['type'] in ['deflection', 'normal']:
            # First create a record with just the state features and action
            # Reward will be computed after merging with receiver logs
            action = 1 if event['type'] == 'deflection' else 0
            
            # Get fw_port_depth based on event type
            if event['type'] == 'deflection':
                fw_port_depth = event['fw_port_depth']
            else:  # Normal event
                port = event['port']
                # Try to get fw_port_depth from the stored values
                fw_port_depth = fw_port_depths.get(port, 0)
            
            # Create data entry with TCP information instead of flow_id and seq
            entry = {
                'timestamp': event['timestamp_str'],
                'action': action,
                'reward': 0,  # Placeholder, will be computed later
                'packet_size': packet_size,
                'total_queue_depth': total_queue_depth,
                'fw_port_depth': fw_port_depth
            }
            
            # Add TCP information if available
            if current_tcp['src_ip']:
                entry['src_ip'] = current_tcp['src_ip']
                entry['dst_ip'] = current_tcp['dst_ip']
                entry['src_port'] = current_tcp['src_port']
                entry['dst_port'] = current_tcp['dst_port']
                entry['tcp_seq'] = current_tcp['tcp_seq']
            
            dataset.append(entry)
    
    # Write to CSV with updated headers
    with open(output_csv, 'w', newline='') as csv_out:
        headers = ['timestamp', 'action', 'reward', 'total_queue_depth', 'fw_port_depth',
                  'packet_size', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'tcp_seq']
                  
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

def add_out_of_order_detection(exp_dir):
    """
    Process server pcap files to identify TCP out-of-order packets and create CSV files
    with out-of-order flags.
    
    Args:
        exp_dir: Path to the experiment directory containing pcap files
        
    Returns:
        List of paths to the generated out-of-order data CSV files
    """
    import sys
    sys.path.append('/home/ubuntu/p4burst')
    from retr_flag_processing import add_out_of_order_flags
    
    print(f"Analyzing TCP out-of-order packets in experiment: {exp_dir}")
    
    # Call the implementation from retr_flag_processing.py
    output_files = add_out_of_order_flags(exp_dir)
    
    if output_files:
        print(f"Successfully created {len(output_files)} out-of-order data files")
        for file in output_files:
            print(f" - {file}")
    else:
        print("No out-of-order data files were created")
    
    return output_files

def merge_switch_logs_with_retr_data(switch_datasets, retr_data_files, output_dir):
    """
    Merge switch log datasets with out-of-order data files from server pcaps.
    This adds out-of-order flags to switch log entries based on matching TCP info.
    
    Args:
        switch_datasets: List of paths to switch log CSV files
        retr_data_files: List of paths to out-of-order data CSV files
        output_dir: Directory to save merged output files
        
    Returns:
        List of paths to the merged dataset files
    """
    print(f"Merging switch logs with out-of-order data")
    
    if not switch_datasets:
        print("Error: No switch datasets provided")
        return []
        
    if not retr_data_files:
        print("Warning: No out-of-order data files provided, using switch logs as-is")
        return switch_datasets
    
    # Load all out-of-order data into a single DataFrame
    retr_df = None
    for retr_file in retr_data_files:
        try:
            df = pd.read_csv(retr_file)
            if retr_df is None:
                retr_df = df
            else:
                retr_df = pd.concat([retr_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading out-of-order file {retr_file}: {e}")
    
    if retr_df is None or retr_df.empty:
        print("Error: Could not load any out-of-order data")
        return []
    
    # Create a lookup dictionary for quick out-of-order flag retrieval
    # Key: (src_ip, dst_ip, src_port, dst_port, tcp_seq)
    retr_lookup = {}
    for _, row in retr_df.iterrows():
        key = (
            row['src_ip'], 
            row['dst_ip'], 
            int(row['src_port']), 
            int(row['dst_port']), 
            int(row['tcp_seq'])
        )
        retr_lookup[key] = int(row['out_of_order_flag'])
    
    print(f"Loaded {len(retr_lookup)} unique TCP packets with out-of-order data")
    
    # Process each switch dataset
    merged_datasets = []
    for i, switch_csv in enumerate(switch_datasets):
        try:
            print(f"Processing switch dataset {i+1}/{len(switch_datasets)}: {switch_csv}")
            
            # Read switch dataset
            switch_df = pd.read_csv(switch_csv)
            
            if switch_df.empty:
                print(f"Warning: Switch dataset {switch_csv} is empty")
                continue
                
            # Check if the necessary TCP columns are present
            required_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'tcp_seq']
            if not all(col in switch_df.columns for col in required_cols):
                print(f"Warning: Switch dataset {switch_csv} missing required TCP columns")
                continue
            
            # Add out_of_order_flag column with default value 0
            switch_df['out_of_order_flag'] = 0
            
            # Match entries with out-of-order data
            found_count = 0
            
            for idx, row in switch_df.iterrows():
                # Create lookup key
                try:
                    key = (
                        row['src_ip'],
                        row['dst_ip'],
                        int(row['src_port']),
                        int(row['dst_port']),
                        int(row['tcp_seq'])
                    )
                    
                    # Look up out-of-order flag
                    if key in retr_lookup:
                        switch_df.at[idx, 'out_of_order_flag'] = retr_lookup[key]
                        found_count += 1
                except (KeyError, ValueError, TypeError) as e:
                    # Skip entries with missing or invalid TCP info
                    continue
            
            # Initialize RTT column with a default value (will be updated later)
            switch_df['rtt'] = None
            
            # Save merged dataset
            output_file = os.path.join(output_dir, f"s{i+1}_with_oo.csv")
            switch_df.to_csv(output_file, index=False)
            
            print(f"Found {found_count} matching packets with out-of-order data")
            print(f"Saved merged dataset to {output_file}")
            
            merged_datasets.append(output_file)
            
        except Exception as e:
            print(f"Error processing switch dataset {switch_csv}: {e}")
            traceback.print_exc()
    
    print(f"Merged {len(merged_datasets)} switch datasets with out-of-order data")
    return merged_datasets

def extract_rtt_using_tshark(pcap_file, output_csv=None):
    """
    Extract RTT measurements from a client PCAP file using tshark's built-in RTT analysis.
    
    Args:
        pcap_file: Path to the client PCAP file
        output_csv: Optional path to write packet RTT data
        
    Returns:
        Dictionary mapping flow IDs to list of RTT measurements
    """
    import pandas as pd
    
    # If output_csv not specified, create one based on pcap filename
    if output_csv is None:
        output_csv = pcap_file.replace('.pcap', '_rtt.csv')
    
    # Extract client IP from filename
    filename = os.path.basename(pcap_file)
    client_ip = None
    try:
        # Format is typically 'bg_client_10.0.1.1_12345.pcap' or 'bursty_client_10.0.1.1.pcap'
        parts = filename.split('_')
        if len(parts) >= 3:
            client_ip = parts[2]
            if client_ip.endswith('.pcap'):
                client_ip = client_ip[:-5]
    except:
        print(f"Warning: Could not extract client IP from {filename}")
    
    # Run tshark command to extract all packets
    tshark_cmd = [
        'tshark', '-r', pcap_file,
        '-T', 'fields',
        '-e', 'frame.number',
        '-e', 'frame.time_epoch',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'tcp.srcport',
        '-e', 'tcp.dstport',
        '-e', 'tcp.seq',
        '-e', 'tcp.ack',
        '-e', 'ip.len',
        '-e', 'tcp.len',
        '-e', 'tcp.analysis.ack_rtt',
        '-o', 'tcp.relative_sequence_numbers: false',
        '-E', 'header=y',
        '-E', 'separator=,'
    ]
    
    try:
        import subprocess
        import io
        
        # Run tshark and get output
        result = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True)
        
        if result.returncode != 0:
            print(f"Error running tshark: {result.stderr}")
            return {}
        
        # Convert tshark output to DataFrame
        df = pd.read_csv(io.StringIO(result.stdout))
        
        # Rename columns to standardized names
        df.columns = ['frame_number', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 
                      'dst_port', 'tcp_seq', 'tcp_ack', 'packet_size', 'payload_size', 'rtt_analysis'] 
        
        # Convert payload_size to int (with 0 as default for empty values)
        df['payload_size'] = df['payload_size'].fillna(0).astype(int)
        
        # Convert RTT to float where possible, otherwise NaN
        df['rtt_analysis'] = pd.to_numeric(df['rtt_analysis'], errors='coerce')
        
        # Add a flag to identify packets from client vs server
        df['is_from_client'] = df['src_ip'] == client_ip
        
        # Compute minimum RTT of the dataset
        min_rtt = df['rtt_analysis'].min()
        if pd.isna(min_rtt) or min_rtt <= 0:
            min_rtt = 0.001  # Default to 1ms if no valid RTT found
            print(f"No valid RTT found, using default min_rtt = {min_rtt}")
        else:
            print(f"Minimum RTT: {min_rtt}")
        
        # Create a column for final RTT values
        df['rtt'] = None
        
        # Step 1: For client packets with valid RTT, keep it
        mask_client_with_rtt = df['is_from_client'] & df['rtt_analysis'].notna()
        df.loc[mask_client_with_rtt, 'rtt'] = df.loc[mask_client_with_rtt, 'rtt_analysis']
        
        # Step 2: For server ACKs with valid RTT, assign that RTT to the matching client packet
        # First, we assign min_rtt to these ACKs
        mask_server_ack_with_rtt = (~df['is_from_client']) & df['rtt_analysis'].notna()
        df.loc[mask_server_ack_with_rtt, 'rtt'] = min_rtt
        
        # Then, find the corresponding client packets and assign them the RTT from the ACK
        for idx in df[mask_server_ack_with_rtt].index:
            ack_row = df.loc[idx]
            acked_seq = ack_row['tcp_ack']
            rtt_value = ack_row['rtt_analysis']
            
            # Find client packets with matching sequence number and payload > 0
            matching_packets = df[(df['is_from_client']) & 
                                  (df['tcp_seq'] == acked_seq) & 
                                  (df['payload_size'] > 0)]
            
            if not matching_packets.empty:
                # Assign the RTT to all matching packets
                for match_idx in matching_packets.index:
                    df.loc[match_idx, 'rtt'] = rtt_value
        
        # Step 3: For any remaining packets without RTT, assign min_rtt
        df['rtt'] = df['rtt'].fillna(min_rtt)
        
        # Build flow_rtts dictionary for return value
        flow_rtts = {}
        for _, row in df.iterrows():
            flow_id = f"{row['src_ip']}:{row['src_port']}-{row['dst_ip']}:{row['dst_port']}"
            if flow_id not in flow_rtts:
                flow_rtts[flow_id] = []
            flow_rtts[flow_id].append(row['rtt'])
        
        # Save the results to CSV
        output_df = df[['frame_number', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 
                         'tcp_seq', 'packet_size', 'rtt']]
        output_df.to_csv(output_csv, index=False)
        
        print(f"Processed {len(df)} packets")
        print(f"Client packets with RTT: {mask_client_with_rtt.sum()}")
        print(f"Server ACKs with RTT: {mask_server_ack_with_rtt.sum()}")
        print(f"RTT data written to {output_csv}")
        
        return flow_rtts
        
    except Exception as e:
        print(f"Error processing PCAP file: {e}")
        traceback.print_exc()
        return {}

def add_rtt_to_switch_logs(switch_datasets, exp_dir, output_dir):
    """
    Add RTT measurements to switch logs from both background and bursty RTT files in a single pass.
    
    Args:
        switch_datasets: List of CSV files with switch logs
        exp_dir: Experiment directory containing RTT files
        output_dir: Directory to write output files
    
    Returns:
        List of new dataset files with RTT information added
    """
    import pandas as pd
    import os
    import traceback
    import numpy as np
    
    print(f"Adding RTT measurements to switch logs...")
    
    # If no switch datasets provided, return empty list
    if not switch_datasets:
        print("No switch datasets provided.")
        return []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find background and bursty RTT files
    bg_rtt_files = []
    bursty_rtt_files = []
    
    # Find all RTT CSV files in the experiment directory
    for root, _, files in os.walk(exp_dir):
        for file in files:
            if file.startswith('rtt_bg_') and file.endswith('.csv'):
                bg_rtt_files.append(os.path.join(root, file))
            elif file.startswith('rtt_bursty_') and file.endswith('.csv'):
                bursty_rtt_files.append(os.path.join(root, file))
    
    print(f"Found {len(bg_rtt_files)} background RTT files and {len(bursty_rtt_files)} bursty RTT files")
    
    # Dictionary to store output files
    output_files = []
    
    # Process each switch dataset
    for dataset_file in switch_datasets:
        try:
            # Create output filename
            basename = os.path.basename(dataset_file)
            output_file = os.path.join(output_dir, basename.replace('_oo.csv', '_with_rtt.csv'))
            
            # Read switch dataset
            switch_df = pd.read_csv(dataset_file)
            
            # Skip empty files
            if switch_df.empty:
                print(f"Skipping empty switch dataset: {dataset_file}")
                continue
            
            # Process background RTT files if available
            if bg_rtt_files:
                switch_df = process_background_rtt(switch_df, bg_rtt_files, basename)
            
            # Process bursty RTT files if available
            if bursty_rtt_files:
                switch_df = process_bursty_rtt(switch_df, bursty_rtt_files, basename)
            
            # Save to output file
            switch_df.to_csv(output_file, index=False)
            print(f"Saved combined RTT data to {output_file}")
            
            output_files.append(output_file)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_file}: {e}")
            traceback.print_exc()
    
    if not output_files:
        print("No output files created, returning original datasets")
        return switch_datasets
    
    return output_files

def process_background_rtt(switch_df, bg_rtt_files, basename):
    """
    Process background RTT files and add RTT measurements to switch dataframe.
    
    Args:
        switch_df: DataFrame with switch logs
        bg_rtt_files: List of background RTT measurement files
        basename: Base name of the switch dataset file (for logging)
    
    Returns:
        Updated DataFrame with background RTT information added
    """
    import pandas as pd
    import traceback
    
    print(f"Processing background RTT for {basename}...")
    
    # Load all RTT dataframes
    rtt_dfs = []
    for rtt_file in bg_rtt_files:
        try:
            df = pd.read_csv(rtt_file)
            # Skip empty files
            if df.empty:
                print(f"Skipping empty RTT file: {rtt_file}")
                continue
                
            # Add to list of dataframes
            rtt_dfs.append(df)
        except Exception as e:
            print(f"Error reading RTT file {rtt_file}: {e}")
    
    # Combine all RTT dataframes
    if not rtt_dfs:
        print("No valid background RTT data found")
        return switch_df
        
    rtt_df = pd.concat(rtt_dfs, ignore_index=True)
    print(f"Loaded {len(rtt_df)} total background RTT measurements")
    
    # Create unique keys for each packet in the RTT data
    rtt_df['flow_key'] = rtt_df['src_ip'] + ':' + rtt_df['dst_ip'] + ':' + \
                         rtt_df['src_port'].astype(str) + ':' + rtt_df['dst_port'].astype(str)
    rtt_df['seq_key'] = rtt_df['tcp_seq'].astype(str) + ':' + rtt_df['packet_size'].astype(str)
    rtt_df['full_key'] = rtt_df['flow_key'] + ':' + rtt_df['seq_key']
    
    # Create a dictionary for fast lookup of RTT values
    rtt_dict = {}
    for _, row in rtt_df.iterrows():
        rtt_dict[row['full_key']] = row['rtt']
    
    # Standard column names for switch logs
    columns = {
        'packet_size': 'packet_size',
        'src_ip': 'src_ip',
        'dst_ip': 'dst_ip',
        'src_port': 'src_port',
        'dst_port': 'dst_port',
        'tcp_seq': 'tcp_seq'
    }
    
    # Try to map standard column names to actual names in the dataframe
    for std_col, actual_col in columns.items():
        if actual_col not in switch_df.columns:
            # Try to find the column using common patterns
            if std_col == 'packet_size' and any(col in switch_df.columns for col in ['length', 'size', 'len']):
                for alt in ['length', 'size', 'len']:
                    if alt in switch_df.columns:
                        columns[std_col] = alt
                        break
            elif std_col == 'src_ip' and 'source_ip' in switch_df.columns:
                columns[std_col] = 'source_ip'
            elif std_col == 'dst_ip' and 'destination_ip' in switch_df.columns:
                columns[std_col] = 'destination_ip'
            elif std_col == 'src_port' and 'sport' in switch_df.columns:
                columns[std_col] = 'sport'
            elif std_col == 'dst_port' and 'dport' in switch_df.columns:
                columns[std_col] = 'dport'
            elif std_col == 'tcp_seq' and any(col in switch_df.columns for col in ['seq', 'sequence']):
                for alt in ['seq', 'sequence']:
                    if alt in switch_df.columns:
                        columns[std_col] = alt
                        break
            else:
                # Use positional fallbacks
                if std_col == 'packet_size':
                    columns[std_col] = switch_df.columns[5]
                elif std_col == 'src_ip':
                    columns[std_col] = switch_df.columns[6]
                elif std_col == 'dst_ip':
                    columns[std_col] = switch_df.columns[7]
                elif std_col == 'src_port':
                    columns[std_col] = switch_df.columns[8]
                elif std_col == 'dst_port':
                    columns[std_col] = switch_df.columns[9]
                elif std_col == 'tcp_seq':
                    columns[std_col] = switch_df.columns[10]
    
    # Create flow and sequence keys to match with RTT data
    switch_df['flow_key'] = switch_df[columns['src_ip']].astype(str) + ':' + \
                           switch_df[columns['dst_ip']].astype(str) + ':' + \
                           switch_df[columns['src_port']].astype(str) + ':' + \
                           switch_df[columns['dst_port']].astype(str)
    
    # Convert packet_size to int with 0 for empty values
    switch_df[columns['packet_size']] = pd.to_numeric(
        switch_df[columns['packet_size']], errors='coerce').fillna(0).astype(int)
    
    # Convert tcp_seq to string
    switch_df[columns['tcp_seq']] = switch_df[columns['tcp_seq']].astype(str)
    
    # Create sequence key
    switch_df['seq_key'] = switch_df[columns['tcp_seq']] + ':' + \
                          switch_df[columns['packet_size']].astype(str)
    
    # Create full key
    switch_df['full_key'] = switch_df['flow_key'] + ':' + switch_df['seq_key']
    
    # Add rtt column if it doesn't exist
    if 'rtt' not in switch_df.columns:
        switch_df['rtt'] = None
    
    # Add rtt_source column to track where RTT came from
    if 'rtt_source' not in switch_df.columns:
        switch_df['rtt_source'] = None
    
    # Count matches
    matches = 0
    small_packet_matches = 0
    seq_only_matches = 0
    
    # Find RTT for each packet in the switch log
    for idx, row in switch_df.iterrows():
        # Skip if already has an RTT value
        if not pd.isna(switch_df.at[idx, 'rtt']):
            continue
            
        # Try exact match first
        if row['full_key'] in rtt_dict:
            switch_df.at[idx, 'rtt'] = rtt_dict[row['full_key']]
            switch_df.at[idx, 'rtt_source'] = 'bg_exact'
            matches += 1
        else:
            # If packet is small, try to find any match with same sequence but different size
            if row[columns['packet_size']] < 1448:
                flow_prefix = row['flow_key'] + ':' + row[columns['tcp_seq']] + ':'
                for key, value in rtt_dict.items():
                    if key.startswith(flow_prefix) and key != row['full_key']:
                        # Check if the other key also has a small packet size
                        packet_size = int(key.split(':')[-1])
                        if packet_size < 1448:
                            switch_df.at[idx, 'rtt'] = value
                            switch_df.at[idx, 'rtt_source'] = 'bg_small_pkt'
                            small_packet_matches += 1
                            break
            
            # If still no match, try just the sequence number
            if pd.isna(switch_df.at[idx, 'rtt']):
                flow_prefix = row['flow_key'] + ':' + row[columns['tcp_seq']] + ':'
                for key, value in rtt_dict.items():
                    if key.startswith(flow_prefix):
                        switch_df.at[idx, 'rtt'] = value
                        switch_df.at[idx, 'rtt_source'] = 'bg_seq_only'
                        seq_only_matches += 1
                        break
    
    print(f"Background RTT matches for {basename}:")
    print(f"  - Exact matches: {matches}")
    print(f"  - Small packet matches: {small_packet_matches}")
    print(f"  - Sequence-only matches: {seq_only_matches}")
    print(f"  - Total matched: {matches + small_packet_matches + seq_only_matches}")
    print(f"  - Remaining without RTT: {switch_df['rtt'].isna().sum()}")
    
    return switch_df

def process_bursty_rtt(switch_df, bursty_rtt_files, basename):
    """
    Process bursty RTT files and add RTT measurements to switch dataframe.
    Uses a combination of flow, timestamp ordering, and packet size to match packets,
    as bursty server responses often have identical sequence numbers.
    
    Args:
        switch_df: DataFrame with switch logs
        bursty_rtt_files: List of bursty RTT measurement files
        basename: Base name of the switch dataset file (for logging)
    
    Returns:
        Updated DataFrame with bursty RTT information added
    """
    import pandas as pd
    import traceback
    import numpy as np
    
    print(f"Processing bursty RTT for {basename}...")
    
    # Load all RTT dataframes
    rtt_dfs = []
    for rtt_file in bursty_rtt_files:
        try:
            df = pd.read_csv(rtt_file)
            # Skip empty files
            if df.empty:
                print(f"Skipping empty RTT file: {rtt_file}")
                continue
                
            # Add to list of dataframes
            rtt_dfs.append(df)
        except Exception as e:
            print(f"Error reading RTT file {rtt_file}: {e}")
    
    # Combine all RTT dataframes
    if not rtt_dfs:
        print("No valid bursty RTT data found")
        return switch_df
        
    rtt_df = pd.concat(rtt_dfs, ignore_index=True)
    print(f"Loaded {len(rtt_df)} total bursty RTT measurements")
    
    # Find the RTT column name - could be 'rtt' or 'final_rtt'
    rtt_column = None
    for col_name in ['rtt', 'final_rtt']:
        if col_name in rtt_df.columns:
            rtt_column = col_name
            break
            
    if not rtt_column:
        print("Error: No RTT column found in bursty RTT data")
        return switch_df
    
    # Extract client IPs from the RTT files
    client_ips = set()
    for file in bursty_rtt_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 3:
            client_ip = parts[2]
            if '.' in client_ip:  # Simple validation that it looks like an IP
                if client_ip.endswith('.csv'):
                    client_ip = client_ip[:-4]
                client_ips.add(client_ip)
    
    if not client_ips:
        print("Warning: Could not extract any client IPs from RTT filenames")
    
    # Identify server response packets (from server to client with payload)
    # First determine if payload column is present and what it's called
    payload_column = None
    for col_name in ['payload_size', 'tcp_len']:
        if col_name in rtt_df.columns:
            payload_column = col_name
            break
    
    if payload_column:
        # Mark packets going to client with payload as server responses
        rtt_df['is_server_response'] = False
        for client_ip in client_ips:
            mask = (rtt_df['dst_ip'] == client_ip) & (rtt_df[payload_column] > 0)
            rtt_df.loc[mask, 'is_server_response'] = True
            
        if rtt_df['is_server_response'].sum() == 0:
            # Try the reverse if no matches (maybe client/server roles got swapped)
            for client_ip in client_ips:
                mask = (rtt_df['src_ip'] == client_ip) & (rtt_df[payload_column] > 0)
                rtt_df.loc[mask, 'is_server_response'] = True
            
            if rtt_df['is_server_response'].sum() == 0:
                # If still no matches, just mark all packets with payload as server responses
                print("Warning: Could not identify server responses by client IPs, using payload presence")
                rtt_df['is_server_response'] = rtt_df[payload_column] > 0
    else:
        # If we can't determine payload, use IP information if available
        if 'is_from_server' in rtt_df.columns:
            rtt_df['is_server_response'] = rtt_df['is_from_server']
        else:
            # Mark all packets as potential server responses as fallback
            print("Warning: Cannot identify server responses reliably - treating all packets as potential matches")
            rtt_df['is_server_response'] = True
    
    print(f"Identified {rtt_df['is_server_response'].sum()} server response packets in RTT data")
    
    # Group RTT data by flow
    rtt_df['flow_key'] = rtt_df['src_ip'] + ':' + rtt_df['dst_ip'] + ':' + \
                        rtt_df['src_port'].astype(str) + ':' + rtt_df['dst_port'].astype(str)
    
    # Standard column names for switch logs
    columns = {
        'packet_size': 'packet_size',
        'src_ip': 'src_ip',
        'dst_ip': 'dst_ip',
        'src_port': 'src_port',
        'dst_port': 'dst_port',
        'tcp_seq': 'tcp_seq'
    }
    
    # Map column names to actual names in the dataframe
    for std_col, actual_col in columns.items():
        if actual_col not in switch_df.columns:
            # Try to find the column using common patterns
            if std_col == 'packet_size' and any(col in switch_df.columns for col in ['length', 'size', 'len']):
                for alt in ['length', 'size', 'len']:
                    if alt in switch_df.columns:
                        columns[std_col] = alt
                        break
            elif std_col == 'src_ip' and 'source_ip' in switch_df.columns:
                columns[std_col] = 'source_ip'
            elif std_col == 'dst_ip' and 'destination_ip' in switch_df.columns:
                columns[std_col] = 'destination_ip'
            elif std_col == 'src_port' and 'sport' in switch_df.columns:
                columns[std_col] = 'sport'
            elif std_col == 'dst_port' and 'dport' in switch_df.columns:
                columns[std_col] = 'dport'
            elif std_col == 'tcp_seq' and any(col in switch_df.columns for col in ['seq', 'sequence']):
                for alt in ['seq', 'sequence']:
                    if alt in switch_df.columns:
                        columns[std_col] = alt
                        break
    
    # Check if required columns are available
    missing_cols = [col for col in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'tcp_seq'] 
                   if columns.get(col) is None]
    if missing_cols:
        print(f"Warning: Missing required columns in {basename}: {missing_cols}")
        print("Using fallback column detection...")
        
        # Fallback - try to infer columns by position
        col_mapping = {
            'src_ip': 'src_ip', 
            'dst_ip': 'dst_ip',
            'src_port': 'src_port', 
            'dst_port': 'dst_port',
            'tcp_seq': 'tcp_seq'
        }
        
        # Check if columns exist by name
        for col in missing_cols:
            if col_mapping[col] in switch_df.columns:
                columns[col] = col_mapping[col]
        
        # If still missing, try positional fallbacks
        still_missing = [col for col in missing_cols if columns.get(col) is None]
        if still_missing:
            print(f"Still missing required columns: {still_missing}")
            if len(switch_df.columns) >= 11:  # If enough columns exist
                if 'src_ip' in still_missing:
                    columns['src_ip'] = switch_df.columns[6]
                if 'dst_ip' in still_missing:
                    columns['dst_ip'] = switch_df.columns[7]
                if 'src_port' in still_missing:
                    columns['src_port'] = switch_df.columns[8]
                if 'dst_port' in still_missing:
                    columns['dst_port'] = switch_df.columns[9]
                if 'tcp_seq' in still_missing:
                    columns['tcp_seq'] = switch_df.columns[10]
            else:
                print(f"Not enough columns in dataset (only {len(switch_df.columns)}) - skipping")
                return switch_df
    
    # Create flow key in switch dataset if not already done
    if 'flow_key' not in switch_df.columns:
        switch_df['flow_key'] = switch_df[columns['src_ip']].astype(str) + ':' + \
                               switch_df[columns['dst_ip']].astype(str) + ':' + \
                               switch_df[columns['src_port']].astype(str) + ':' + \
                               switch_df[columns['dst_port']].astype(str)
    
    # Add rtt column if it doesn't exist
    if 'rtt' not in switch_df.columns:
        switch_df['rtt'] = None
    
    # Add rtt_source column to track where RTT came from
    if 'rtt_source' not in switch_df.columns:
        switch_df['rtt_source'] = None
    
    # Match counts
    match_count = 0
    flow_avg_matches = 0
    
    # Process each flow in the switch dataset
    for flow in switch_df['flow_key'].unique():
        # Get RTT data for this flow
        flow_rtt = rtt_df[rtt_df['flow_key'] == flow]
        
        if flow_rtt.empty:
            continue
        
        # Get switch data for this flow
        flow_switch = switch_df[switch_df['flow_key'] == flow]
        
        # For each sequence number in this flow
        for seq in flow_switch[columns['tcp_seq']].unique():
            # Get switch packets with this sequence
            seq_switch = flow_switch[flow_switch[columns['tcp_seq']] == seq]
            
            # Skip packets that already have RTT values from background processing
            # unless there are server responses we can match better
            seq_switch_needing_rtt = seq_switch[seq_switch['rtt'].isna() | 
                                               (seq_switch['rtt_source'].isin(['bg_small_pkt', 'bg_seq_only']))]
            
            if seq_switch_needing_rtt.empty:
                continue
            
            if len(seq_switch_needing_rtt) <= 1:
                # If only one packet with this sequence, try direct match
                seq_rtt = flow_rtt[flow_rtt['tcp_seq'] == seq]
                
                if not seq_rtt.empty:
                    # Just use the first RTT value if multiple exist
                    rtt_value = seq_rtt.iloc[0][rtt_column]
                    if not pd.isna(rtt_value):
                        for idx in seq_switch_needing_rtt.index:
                            switch_df.at[idx, 'rtt'] = rtt_value
                            switch_df.at[idx, 'rtt_source'] = 'bursty_exact'
                            match_count += 1
            else:
                # Multiple packets with same sequence - use positional matching
                seq_rtt = flow_rtt[flow_rtt['tcp_seq'] == seq]
                
                if seq_rtt.empty:
                    continue
                
                # Focus on server responses if possible
                server_resp = seq_rtt[seq_rtt['is_server_response']]
                if not server_resp.empty:
                    seq_rtt = server_resp
                
                # If we have packet sizes in both datasets, try matching by size
                if columns.get('packet_size') and 'packet_size' in seq_rtt.columns:
                    for idx, switch_row in seq_switch_needing_rtt.iterrows():
                        switch_size = switch_row[columns['packet_size']]
                        
                        # Find closest match by packet size
                        closest = None
                        min_diff = float('inf')
                        
                        for _, rtt_row in seq_rtt.iterrows():
                            rtt_size = rtt_row['packet_size']
                            diff = abs(switch_size - rtt_size)
                            
                            if diff < min_diff:
                                min_diff = diff
                                closest = rtt_row
                        
                        # If we found a reasonable match (size difference < 100 bytes)
                        if closest is not None and min_diff < 100:
                            rtt_value = closest[rtt_column]
                            if not pd.isna(rtt_value):
                                switch_df.at[idx, 'rtt'] = rtt_value
                                switch_df.at[idx, 'rtt_source'] = 'bursty_size'
                                match_count += 1
                
                # If we still have unmatched packets, use positional ordering
                unmatched = seq_switch_needing_rtt[switch_df.loc[seq_switch_needing_rtt.index, 'rtt'].isna()]
                
                if not unmatched.empty and len(seq_rtt) > 0:
                    # Sort RTT packets (by frame number if available, otherwise by index)
                    if 'frame_number' in seq_rtt.columns:
                        seq_rtt_sorted = seq_rtt.sort_values('frame_number')
                    else:
                        seq_rtt_sorted = seq_rtt.sort_index()
                    
                    # Sort switch packets by index
                    unmatched_sorted = unmatched.sort_index()
                    
                    # Assign RTTs based on relative position
                    for i, (idx, _) in enumerate(unmatched_sorted.iterrows()):
                        if i < len(seq_rtt_sorted):
                            rtt_value = seq_rtt_sorted.iloc[i][rtt_column]
                            if not pd.isna(rtt_value):
                                switch_df.at[idx, 'rtt'] = rtt_value
                                switch_df.at[idx, 'rtt_source'] = 'bursty_position'
                                match_count += 1
    
    # For any remaining unmatched packets, use averages from their flow
    unmatched = switch_df[switch_df['rtt'].isna()]
    
    if not unmatched.empty:
        # Compute average RTT per flow
        flow_rtts = {}
        for flow in rtt_df['flow_key'].unique():
            flow_data = rtt_df[rtt_df['flow_key'] == flow]
            rtts = flow_data[rtt_column].dropna()
            if not rtts.empty:
                flow_rtts[flow] = rtts.median()
        
        # Apply flow averages to unmatched packets
        flow_avg_matches = 0
        for idx, row in unmatched.iterrows():
            if row['flow_key'] in flow_rtts:
                switch_df.at[idx, 'rtt'] = flow_rtts[row['flow_key']]
                switch_df.at[idx, 'rtt_source'] = 'bursty_flow_avg'
                flow_avg_matches += 1
        
        print(f"Assigned flow average RTT to {flow_avg_matches} packets")
    
    # We do NOT fill with min_rtt here, as requested
    
    print(f"Bursty RTT matches for {basename}:")
    print(f"  - Direct matches: {match_count}")
    print(f"  - Flow average matches: {flow_avg_matches}")
    print(f"  - Remaining without RTT: {switch_df['rtt'].isna().sum()}")
    
    return switch_df

def create_final_datasets(retr_datasets, rtt_datasets, output_dir):
    """
    Create final datasets by merging all data from retransmission and RTT datasets.
    
    Args:
        retr_datasets: List of CSV files with retransmission information
        rtt_datasets: List of CSV files with RTT information
        output_dir: Directory to write final datasets
    
    Returns:
        List of final dataset files
    """
    import pandas as pd
    import os
    import re
    
    # Dictionary to store final dataset files
    final_datasets = []
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create mapping between switch IDs and retransmission datasets
    retr_map = {}
    for retr_file in retr_datasets:
        # Extract switch ID from filename
        match = re.search(r's(\d+)_', os.path.basename(retr_file))
        if match:
            switch_id = 's' + match.group(1)
            retr_map[switch_id] = retr_file
    
    # Create mapping between switch IDs and RTT datasets
    rtt_map = {}
    for rtt_file in rtt_datasets:
        # Extract switch ID from filename
        match = re.search(r's(\d+)_', os.path.basename(rtt_file))
        if match:
            switch_id = 's' + match.group(1)
            rtt_map[switch_id] = rtt_file
    
    # Merge datasets for each switch
    for switch_id in list(set(list(retr_map.keys()) + list(rtt_map.keys()))):
        # Create output filename
        output_file = os.path.join(output_dir, f"{switch_id}_final_dataset.csv")
        
        try:
            # If we have both retransmission and RTT data for this switch
            if switch_id in retr_map and switch_id in rtt_map:
                # Read retransmission data
                retr_df = pd.read_csv(retr_map[switch_id])
                
                # Read RTT data
                rtt_df = pd.read_csv(rtt_map[switch_id])
                
                # Merge datasets
                merged_df = pd.merge(retr_df, rtt_df, how='outer')
                
                # Fill missing values for retransmission column with 0 (no retransmission)
                if 'retransmission' in merged_df.columns:
                    merged_df['retransmission'] = merged_df['retransmission'].fillna(0)
                
                # Fill missing RTT values with minimum RTT
                if 'rtt' in merged_df.columns and merged_df['rtt'].isna().any():
                    min_rtt = merged_df['rtt'].dropna().min()
                    if pd.isna(min_rtt) or min_rtt <= 0:
                        min_rtt = 0.001  # Default to 1ms if no valid RTT found
                    
                    missing_count = merged_df['rtt'].isna().sum()
                    merged_df['rtt'] = merged_df['rtt'].fillna(min_rtt)
                    print(f"Filled {missing_count} missing RTT values with {min_rtt} in final dataset {switch_id}")
                
            # If we only have retransmission data
            elif switch_id in retr_map:
                merged_df = pd.read_csv(retr_map[switch_id])
                
                # Add empty RTT column
                if 'rtt' not in merged_df.columns:
                    merged_df['rtt'] = 0.001  # Default to 1ms
                    print(f"No RTT data for {switch_id}, using default 1ms RTT")
            
            # If we only have RTT data
            elif switch_id in rtt_map:
                merged_df = pd.read_csv(rtt_map[switch_id])
                
                # Add empty retransmission column
                if 'retransmission' not in merged_df.columns:
                    merged_df['retransmission'] = 0
                    print(f"No retransmission data for {switch_id}, assuming no retransmissions")
                
                # Fill missing RTT values with minimum RTT
                if 'rtt' in merged_df.columns and merged_df['rtt'].isna().any():
                    min_rtt = merged_df['rtt'].dropna().min()
                    if pd.isna(min_rtt) or min_rtt <= 0:
                        min_rtt = 0.001  # Default to 1ms if no valid RTT found
                    
                    missing_count = merged_df['rtt'].isna().sum()
                    merged_df['rtt'] = merged_df['rtt'].fillna(min_rtt)
                    print(f"Filled {missing_count} missing RTT values with {min_rtt} in final dataset {switch_id}")
            
            # Save final dataset
            merged_df.to_csv(output_file, index=False)
            final_datasets.append(output_file)
            
            print(f"Created final dataset for {switch_id} with {len(merged_df)} rows")
            
        except Exception as e:
            print(f"Error creating final dataset for {switch_id}: {e}")
    
    return final_datasets

def extract_response_rtt_from_pcap(pcap_file, output_csv=None):
    """
    Extract RTT measurements for server responses in bursty TCP traffic.
    Maps RTTs from client ACKs to the server responses they acknowledge.
    
    Args:
        pcap_file: Path to the client PCAP file
        output_csv: Optional path to write response RTT data
        
    Returns:
        Dictionary mapping flow IDs to list of RTT measurements
    """
    import pandas as pd
    import subprocess
    import os
    import traceback
    
    print(f"Extracting response RTT from {pcap_file}")
    
    # If output_csv not specified, create one based on pcap filename
    if output_csv is None:
        output_csv = pcap_file.replace('.pcap', '_response_rtt.csv')
    
    # Extract client IP from filename
    filename = os.path.basename(pcap_file)
    client_ip = None
    try:
        # Format is typically 'bursty_client_10.0.1.1.pcap'
        parts = filename.split('_')
        if len(parts) >= 3:
            client_ip = parts[2]
            if client_ip.endswith('.pcap'):
                client_ip = client_ip[:-5]
    except:
        print(f"Warning: Could not extract client IP from {filename}")
    
    # Run tshark command to extract all packets, including TCP flags
    tshark_cmd = [
        'tshark', '-r', pcap_file,
        '-T', 'fields',
        '-e', 'frame.number',
        '-e', 'frame.time_epoch',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'tcp.srcport',
        '-e', 'tcp.dstport',
        '-e', 'tcp.seq',
        '-e', 'tcp.ack',
        '-e', 'ip.len',
        '-e', 'tcp.len',
        '-e', 'tcp.flags',
        '-e', 'tcp.flags.syn',
        '-e', 'tcp.flags.ack',
        '-e', 'tcp.flags.push',
        '-e', 'tcp.flags.fin',
        '-e', 'tcp.analysis.ack_rtt',
        '-o', 'tcp.relative_sequence_numbers: false',
        '-E', 'header=y',
        '-E', 'separator=,'
    ]
    
    try:
        # Run tshark and get output
        result = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True)
        
        if result.returncode != 0:
            print(f"Error running tshark: {result.stderr}")
            return {}
        
        import io
        # Convert tshark output to DataFrame
        df = pd.read_csv(io.StringIO(result.stdout))
        
        # Rename columns to standardized names
        df.columns = ['frame_number', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 
                      'dst_port', 'tcp_seq', 'tcp_ack', 'packet_size', 'tcp_len', 
                      'tcp_flags', 'syn_flag', 'ack_flag', 'push_flag', 'fin_flag',
                      'rtt']
        
        # Convert numeric columns
        numeric_cols = ['frame_number', 'tcp_seq', 'tcp_ack', 'packet_size', 'tcp_len', 
                        'syn_flag', 'ack_flag', 'push_flag', 'fin_flag']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Convert RTT to float
        df['rtt'] = pd.to_numeric(df['rtt'], errors='coerce')
        
        # Add direction flags based on IP addresses
        df['is_from_client'] = df['src_ip'] == client_ip
        df['is_from_server'] = ~df['is_from_client']
        
        # Calculate payload size (tcp.len is the payload size)
        df['payload_size'] = df['tcp_len']
        
        # Add packet type classification
        df['is_syn'] = df['syn_flag'] == 1
        df['is_ack'] = df['ack_flag'] == 1
        df['is_syn_ack'] = (df['syn_flag'] == 1) & (df['ack_flag'] == 1)
        df['is_fin'] = df['fin_flag'] == 1
        df['is_push'] = df['push_flag'] == 1
        df['is_data'] = df['payload_size'] > 0
        
        # Print diagnostic information
        print(f"Total packets: {len(df)}")
        print(f"Client packets: {df['is_from_client'].sum()}")
        print(f"Server packets: {df['is_from_server'].sum()}")
        print(f"Server packets with payload > 0: {df[df['is_from_server'] & (df['payload_size'] > 0)].shape[0]}")
        print(f"Client ACKs: {df[df['is_from_client'] & df['is_ack']].shape[0]}")
        print(f"Packets with RTT values: {df['rtt'].notna().sum()}")
        
        # Initialize final RTT column - this will hold our computed RTTs
        df['final_rtt'] = None
        
        # First, preserve any original RTT values that exist
        df.loc[df['rtt'].notna(), 'final_rtt'] = df.loc[df['rtt'].notna(), 'rtt']
        
        # Step 1: Build a mapping from client ACKs with RTT to the server packets they acknowledge
        # First, create a dictionary mapping ACK number to client ACK details
        ack_to_rtt = {}
        client_acks_with_rtt = df[(df['is_from_client']) & (df['is_ack']) & (df['rtt'].notna())]
        for _, row in client_acks_with_rtt.iterrows():
            ack_num = int(row['tcp_ack'])
            rtt_value = float(row['rtt'])
            ack_to_rtt[ack_num] = rtt_value
        
        print(f"Found {len(ack_to_rtt)} client ACKs with RTT values")
        
        # Step 2: Process server data packets and assign RTT values from matching client ACKs
        # We'll create a mapping from sequence range to frame number and final RTT value
        seq_to_frame_rtt = {}
        server_data_packets = df[(df['is_from_server']) & (df['payload_size'] > 0)]
        matched_server_packets = 0
        
        for idx, row in server_data_packets.iterrows():
            seq_num = int(row['tcp_seq'])
            payload_size = int(row['payload_size'])
            frame_num = int(row['frame_number'])
            
            # Calculate the end sequence number (next seq) that would be ACKed
            next_seq = seq_num + payload_size
            
            # Check if this sequence number is acknowledged by a client ACK with RTT
            if next_seq in ack_to_rtt:
                df.loc[idx, 'final_rtt'] = ack_to_rtt[next_seq]
                matched_server_packets += 1
                
                # Print debug info for a few matches
                if matched_server_packets <= 5:
                    print(f"Frame {frame_num}: Server packet with seq {seq_num} (payload={payload_size}) matched with client ACK for seq {next_seq}, RTT={ack_to_rtt[next_seq]}")
        
        print(f"Matched {matched_server_packets} server data packets with client ACKs")
        
        # Step 3: For any remaining server data packets without RTT, try to infer from nearby packets
        # First, create a sorted list of packets with assigned RTTs
        packets_with_rtt = df[df['final_rtt'].notna()].sort_values('frame_number')
        
        # Assign RTTs to any unmatched server data packets
        unmatched_server_packets = df[(df['is_from_server']) & (df['payload_size'] > 0) & (df['final_rtt'].isna())]
        inferred_server_packets = 0
        
        if not unmatched_server_packets.empty and not packets_with_rtt.empty:
            for idx, row in unmatched_server_packets.iterrows():
                frame_num = row['frame_number']
                
                # Find the closest packet with RTT (either before or after)
                packets_before = packets_with_rtt[packets_with_rtt['frame_number'] < frame_num]
                packets_after = packets_with_rtt[packets_with_rtt['frame_number'] > frame_num]
                
                if not packets_before.empty:
                    closest_before = packets_before.iloc[-1]
                    before_distance = frame_num - closest_before['frame_number']
                else:
                    before_distance = float('inf')
                    
                if not packets_after.empty:
                    closest_after = packets_after.iloc[0]
                    after_distance = closest_after['frame_number'] - frame_num
                else:
                    after_distance = float('inf')
                
                # Assign RTT from the closest packet
                if before_distance <= after_distance and before_distance != float('inf'):
                    df.loc[idx, 'final_rtt'] = closest_before['final_rtt']
                    inferred_server_packets += 1
                elif after_distance != float('inf'):
                    df.loc[idx, 'final_rtt'] = closest_after['final_rtt']
                    inferred_server_packets += 1
        
        print(f"Inferred RTT for {inferred_server_packets} additional server data packets")
        
        # Step 4: Fill remaining packets with minimum RTT
        # Calculate the minimum RTT value from all available RTT measurements
        min_rtt = df['final_rtt'].dropna().min()
        
        # If no valid RTT found, use a default value
        if pd.isna(min_rtt) or min_rtt <= 0:
            min_rtt = 0.001  # Default to 1ms if no valid RTT found
            print(f"No valid RTT found, using default min_rtt = {min_rtt}")
        else:
            print(f"Using minimum RTT = {min_rtt} for filling missing values")
        
        # Count packets with missing RTT values
        missing_rtt_count = df['final_rtt'].isna().sum()
        
        # Fill missing RTT values with minimum RTT
        df['final_rtt'] = df['final_rtt'].fillna(min_rtt)

        
        # Final statistics
        packets_with_final_rtt = df['final_rtt'].notna().sum()
        print(f"Final statistics:")
        print(f"  - Total packets with RTT: {packets_with_final_rtt}")
        
        # Save to output CSV (with renamed rtt column)
        output_df = df[['frame_number', 'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 
                        'tcp_seq', 'tcp_ack', 'packet_size', 'tcp_flags', 'final_rtt']]
        output_df = output_df.rename(columns={'final_rtt': 'rtt'})
        output_df.to_csv(output_csv, index=False)
        print(f"RTT data written to {output_csv}")
        
        # Build flow_rtts dictionary for return value
        flow_rtts = {}
        for _, row in df.iterrows():
            if pd.isna(row['final_rtt']):
                continue  # Skip packets without RTT (should be none at this point)
                
            flow_id = f"{row['src_ip']}:{row['src_port']}-{row['dst_ip']}:{row['dst_port']}"
            if flow_id not in flow_rtts:
                flow_rtts[flow_id] = []
            flow_rtts[flow_id].append(row['final_rtt'])
        
        return flow_rtts
        
    except Exception as e:
        print(f"Error processing PCAP file: {e}")
        traceback.print_exc()
        return {}

def cleanup_intermediate_files(exp_dir):
    """
    Clean up intermediate CSV files and logs in the experiment directory.
    Preserves important files like queue logger files, app.log, collection_runner.log, and final switch datasets.
    
    Args:
        exp_dir: Path to the experiment directory
    """
    print(f"Cleaning up intermediate files in {exp_dir}")
    
    # Files to preserve
    preserve_patterns = [
        'queue_logger',  # Queue logger files
        'app.log',      # Application log
        'collection_runner.log',  # Collection runner log
        's\d+_final_dataset.csv'  # Final switch datasets
    ]
    
    # Files to remove
    remove_patterns = [
        'rtt_bg_',      # Background client RTT files
        'rtt_bursty_',  # Bursty client RTT files
        's\d+_with_oo.csv',  # Datasets with out-of-order flags
        's\d+_with_rtt.csv', # Datasets with RTT measurements
        'flow_rtts.csv',     # Flow RTT summary
        'bg_client_.*\.csv', # Background client data files
        'bursty_client_.*\.csv', # Bursty client data files
        'server_.*\.csv',   # Server data files
        # 'burst_server_.*\.log', # Burst server data files
        # 'burst_client_.*\.log', # Burst client data files
        # 'bg_server_.*\.log', # Background server data files
        # 'bg_client_.*\.log', # Background client data files
        'bursty_client_.*\.pcap', # Bursty client data files
        'bg_client_.*\.pcap', # Background client data files
        'bursty_server_.*\.pcap', # Bursty server data files
        'bg_server_.*\.pcap' # Background server data files
    ]
    
    # Convert patterns to regex
    preserve_regex = '|'.join(preserve_patterns)
    remove_regex = '|'.join(remove_patterns)
    
    import re
    preserve_pattern = re.compile(preserve_regex)
    remove_pattern = re.compile(remove_regex)
    
    # Count files for statistics
    total_files = 0
    removed_files = 0
    preserved_files = 0
    
    # Process each file in the directory
    for filename in os.listdir(exp_dir):
        total_files += 1
        filepath = os.path.join(exp_dir, filename)
        
        # Skip directories
        if os.path.isdir(filepath):
            continue
            
        # Check if file should be preserved
        if preserve_pattern.search(filename):
            preserved_files += 1
            continue
            
        # Check if file should be removed
        if remove_pattern.search(filename):
            try:
                os.remove(filepath)
                removed_files += 1
            except Exception as e:
                print(f"Warning: Could not remove {filename}: {e}")
        else:
            preserved_files += 1
    
    print(f"Cleanup complete:")
    print(f"  - Total files: {total_files}")
    print(f"  - Removed: {removed_files}")
    print(f"  - Preserved: {preserved_files}")

def process_and_merge_all_data(topology, exp_dir):
    """
    Complete workflow to process all data:
    1. Process out-of-order flags
    2. Collect switch logs
    3. Merge switch logs with out-of-order flags
    4. Compute RTT from client PCAPs (per-packet)
    5. Merge switch logs with per-packet RTT measurements
    6. Create final datasets with both out-of-order flags and RTT
    
    This is the main function to be called from collection_runner.py.
    
    Args:
        topology: The network topology object
        exp_dir: Path to the experiment directory
        
    Returns:
        List of paths to the final dataset files
    """
    print(f"Processing all data for experiment in {exp_dir}")
    
    # Step 1: Process out-of-order flags
    logs_w_oo_flag = add_out_of_order_detection(exp_dir)
    if logs_w_oo_flag:
        print("Successfully created out-of-order data files")
    else:
        print("Warning: No out-of-order data files created")
    
    # Step 2: Collect switch logs
    switch_datasets = collect_switch_logs(topology, exp_dir)
    
    if not switch_datasets:
        print("Warning: No switch datasets found, cannot proceed")
        return []
    
    # Step 3: Merge switch logs with out-of-order flags
    oo_datasets = merge_switch_logs_with_retr_data(switch_datasets, logs_w_oo_flag, exp_dir)
    if not oo_datasets:
        print("Warning: Failed to merge switch logs with out-of-order flags")
        oo_datasets = switch_datasets  # Use original switch datasets as fallback
    
    # Step 4: Extract RTTs from client PCAPs
    print("Processing client PCAP files for RTT data")
    
    # Step 4a: Process background client PCAP files with tshark
    bg_pcap_files = []
    for file in os.listdir(exp_dir):
        if file.startswith('bg_client_') and file.endswith('.pcap'):
            bg_pcap_files.append(os.path.join(exp_dir, file))
    
    if bg_pcap_files:
        print(f"Found {len(bg_pcap_files)} background client PCAP files")
        for pcap_file in bg_pcap_files:
            try:
                # Extract client IP from filename
                filename = os.path.basename(pcap_file)
                parts = filename.split('_')
                client_ip = parts[2] if len(parts) >= 3 else None
                if client_ip and client_ip.endswith('.pcap'):
                    client_ip = client_ip[:-5]
                
                print(f"Processing background client PCAP file for {client_ip}")
                output_csv = os.path.join(exp_dir, f"rtt_bg_{client_ip}.csv")
                extract_rtt_using_tshark(pcap_file, output_csv)
            except Exception as e:
                print(f"Error processing background client PCAP file {pcap_file}: {e}")
                traceback.print_exc()
    else:
        print("Warning: No background client PCAP files found")
    
    # Step 4b: Process bursty client PCAP files to calculate response RTTs
    bursty_pcap_files = []
    for file in os.listdir(exp_dir):
        if file.startswith('bursty_client_') and file.endswith('.pcap'):
            bursty_pcap_files.append(os.path.join(exp_dir, file))
    
    if bursty_pcap_files:
        print(f"Found {len(bursty_pcap_files)} bursty client PCAP files")
        for pcap_file in bursty_pcap_files:
            try:
                # Extract client IP from filename
                filename = os.path.basename(pcap_file)
                parts = filename.split('_')
                client_ip = parts[2] if len(parts) >= 3 else None
                if client_ip and client_ip.endswith('.pcap'):
                    client_ip = client_ip[:-5]
                
                print(f"Processing bursty client PCAP file for {client_ip}")
                output_csv = os.path.join(exp_dir, f"rtt_bursty_{client_ip}.csv")
                extract_response_rtt_from_pcap(pcap_file, output_csv)
            except Exception as e:
                print(f"Error processing bursty client PCAP file {pcap_file}: {e}")
                traceback.print_exc()
    else:
        print("Warning: No bursty client PCAP files found")
    
    # Step 5: Merge switch logs with per-packet RTT measurements
    rtt_datasets = add_rtt_to_switch_logs(oo_datasets, exp_dir, exp_dir)
    if not rtt_datasets:
        print("Warning: Failed to merge switch logs with RTT measurements")
        rtt_datasets = oo_datasets
    
    # Step 6: Create final datasets
    print("Creating final datasets")
    final_datasets = create_final_datasets(oo_datasets, rtt_datasets, exp_dir)
    
    # Step 7: Clean up intermediate files
    #cleanup_intermediate_files(exp_dir)
    
    # Print summary of created files
    print("\nSummary of created files:")
    print(f" - Final merged datasets: {final_datasets}")
    
    return final_datasets

if __name__ == "__main__":

    
    exp_dir = os.path.join('tmp', '20250404_220042')

    # Process client PCAP files to extract RTT measurements
    # client_rtts = process_client_pcaps_for_rtt(exp_dir)
    
    # Collect switch logs and process them
    topology = LeafSpineTopology(
        num_hosts = 4,
        num_leaf = 2,
        num_spine = 2,
        bw = 10,
        latency = 0,
        p4_program='p4src/sd/sd.p4'
    )
    topology.generate_topology()

    process_and_merge_all_data(topology, exp_dir)

    # switch_datasets = collect_switch_logs(topology, exp_dir)
    
    # if not switch_datasets:
    #     print("Warning: No switch datasets found, skipping merge")
    # else:
    #     # Merge switch logs with RTT measurements
    #     merged_datasets = add_rtt_to_switch_logs(switch_datasets, os.path.join(exp_dir, "flow_rtts.csv"), exp_dir)
    #     if not merged_datasets:
    #         print("Warning: No merged datasets created")
    #     else:
    #         print(f"Merged datasets created: {merged_datasets}")

    # # Print summary of created files
    # print("\nSummary of created files:")
    # print(f" - Switch datasets: {switch_datasets}")
    # print(f" - Merged datasets: {merged_datasets}")
