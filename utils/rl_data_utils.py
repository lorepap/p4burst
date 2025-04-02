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
            switch_df['rtt'] = 0.1  # Default RTT value
            
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
    Focus on per-packet data for merging with switch logs.
    
    Args:
        pcap_file: Path to the client PCAP file
        output_csv: Optional path to write packet RTT data
        
    Returns:
        Dictionary mapping flow IDs to list of RTT measurements
    """
    # Determine if this is a bursty client or background client based on filename
    is_bursty = os.path.basename(pcap_file).startswith('bursty_client_')
    client_type = "bursty" if is_bursty else "background"
    
    print(f"Processing {client_type} client PCAP file: {pcap_file}")
    
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
    
    # Run tshark command to extract RTT information
    tshark_cmd = [
        'tshark', '-r', pcap_file,
        '-T', 'fields',
        '-e', 'frame.number',
        '-e', 'ip.src',
        '-e', 'ip.dst',
        '-e', 'tcp.srcport',
        '-e', 'tcp.dstport',
        '-e', 'tcp.seq',
        '-e', 'tcp.ack',
        '-e', 'tcp.analysis.ack_rtt',
        '-e', 'tcp.time_relative',
        '-o', 'tcp.relative_sequence_numbers: false',
        '-E', 'header=y',
        '-E', 'separator=,'
    ]
    
    try:
        import subprocess
        result = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True)
        
        if result.returncode != 0:
            print(f"Error running tshark: {result.stderr}")
            return {}
        
        # Parse the CSV output
        import io
        import csv
        
        # Dictionary to store RTT values by flow
        flow_rtts = {}
        
        # Create a CSV reader from the output
        csv_reader = csv.reader(io.StringIO(result.stdout))
        
        # Get header row
        headers = next(csv_reader)
        
        # Prepare to write filtered data
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = [
                'frame_number', 'src_ip', 'dst_ip', 'src_port', 'dst_port',
                'seq_num', 'ack_num', 'rtt', 'time_relative'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process each row
            packet_count = 0
            rtt_count = 0
            client_to_server_rtt = 0
            server_to_client_rtt = 0
            
            for row in csv_reader:
                if len(row) < 8:  # Ensure we have enough columns
                    continue  # Skip incomplete rows
                
                # Extract fields
                if len(row) >= 9:
                    frame_number = row[0]
                    src_ip = row[1]
                    dst_ip = row[2]
                    src_port = row[3]
                    dst_port = row[4]
                    seq_num = row[5]
                    ack_num = row[6]
                    rtt = row[7]
                    time_relative = row[8] if len(row) > 8 else ""
                else:
                    # Skip rows that don't have enough columns
                    continue
                
                # Skip rows without RTT
                if not rtt or rtt == "":
                    continue
                
                # RTT filtering logic
                if is_bursty:
                    # For bursty clients, include packets in both directions
                    if client_ip:
                        is_client_to_server = (src_ip == client_ip)
                        is_server_to_client = (dst_ip == client_ip)
                        
                        if not (is_client_to_server or is_server_to_client):
                            continue
                        
                        if is_client_to_server:
                            client_to_server_rtt += 1
                        if is_server_to_client:
                            server_to_client_rtt += 1
                else:
                    # For background clients, only include packets from client to server
                    if client_ip and src_ip != client_ip:
                        continue
                
                # Convert RTT from seconds to milliseconds
                try:
                    rtt = float(rtt)
                except ValueError:
                    continue  # Skip if RTT is not a valid number
                
                # Generate flow ID
                flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                
                # Add to flow RTT dictionary
                if flow_id not in flow_rtts:
                    flow_rtts[flow_id] = []
                flow_rtts[flow_id].append(rtt)
                
                # Write this packet to the output CSV
                writer.writerow({
                    'frame_number': frame_number,
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'seq_num': seq_num,
                    'ack_num': ack_num,
                    'rtt': rtt,
                    'time_relative': time_relative
                })
                
                packet_count += 1
                rtt_count += 1
            
        print(f"Processed {packet_count} packets with {rtt_count} valid RTT measurements")
        print(f"RTT data written to {output_csv}")
        
        return flow_rtts
        
    except Exception as e:
        print(f"Error running tshark or processing output: {e}")
        traceback.print_exc()
        return {}

def process_client_pcaps_for_rtt(exp_dir):
    """
    Process all client PCAP files in the experiment directory to extract RTT measurements using tshark.
    
    Args:
        exp_dir: Path to the experiment directory
        
    Returns:
        Dictionary mapping client IPs to their RTT measurements
    """
    print(f"Processing client PCAP files for RTT in {exp_dir}")
    
    # Dictionary to store RTT measurements by client
    client_rtts = {}
    
    # Find all client PCAP files
    pcap_files = []
    for file in os.listdir(exp_dir):
        if file.startswith('bg_client_') and file.endswith('.pcap'):
            pcap_files.append(os.path.join(exp_dir, file))
    
    if not pcap_files:
        print("No client PCAP files found")
        return client_rtts
    
    print(f"Found {len(pcap_files)} client PCAP files")
    
    # Process each PCAP file
    for pcap_file in pcap_files:
        try:
            # Extract client IP from filename
            filename = os.path.basename(pcap_file)
            parts = filename.split('_')
            client_ip = None
            if len(parts) >= 3:
                client_ip = parts[2]
                if client_ip.endswith('.pcap'):
                    client_ip = client_ip[:-5]
            
            if not client_ip:
                print(f"Warning: Could not extract client IP from filename {filename}")
                continue
                
            print(f"Processing PCAP file for client {client_ip}")
            
            # Extract RTT measurements using tshark
            output_csv = os.path.join(exp_dir, f"rtt_{client_ip}.csv")
            flow_rtts = extract_rtt_using_tshark(pcap_file, output_csv)
            
            if flow_rtts:
                client_rtts[client_ip] = flow_rtts
            
        except Exception as e:
            print(f"Error processing client PCAP file {pcap_file}: {e}")
            traceback.print_exc()
    
    return client_rtts

def compute_flow_rtts(exp_dir):
    """
    Process client PCAP files, compute RTT for each flow using tshark, and generate a CSV summary.
    
    Args:
        exp_dir: Path to the experiment directory
        
    Returns:
        Path to the generated CSV file with RTT measurements
    """
    print(f"Computing RTT for flows in {exp_dir} using tshark")
    
    output_csv = os.path.join(exp_dir, "flow_rtts.csv")
    
    # Process all client PCAP files
    client_rtts = process_client_pcaps_for_rtt(exp_dir)
    
    if not client_rtts:
        print("No RTT measurements found")
        return None
    
    # Prepare data for CSV output
    csv_rows = []
    for client_ip, flow_rtts in client_rtts.items():
        for flow_id, rtts in flow_rtts.items():
            if rtts:
                src_ip, dst_ip, src_port, dst_port = None, None, None, None
                
                # Parse flow ID into components
                try:
                    src_dst = flow_id.split('-')
                    if len(src_dst) == 2:
                        src = src_dst[0].split(':')
                        dst = src_dst[1].split(':')
                        if len(src) == 2 and len(dst) == 2:
                            src_ip = src[0]
                            src_port = src[1]
                            dst_ip = dst[0]
                            dst_port = dst[1]
                except Exception as e:
                    print(f"Warning: Could not parse flow ID {flow_id}: {e}")
                
                min_rtt = min(rtts)
                max_rtt = max(rtts)
                avg_rtt = sum(rtts) / len(rtts)
                median_rtt = sorted(rtts)[len(rtts) // 2]
                
                row = {
                    'client_ip': client_ip,
                    'src_ip': src_ip,
                    'src_port': src_port,
                    'dst_ip': dst_ip,
                    'dst_port': dst_port,
                    'min_rtt_ms': round(min_rtt, 4),
                    'avg_rtt_ms': round(avg_rtt, 4),
                    'median_rtt_ms': round(median_rtt, 4),
                    'max_rtt_ms': round(max_rtt, 4),
                    'num_samples': len(rtts)
                }
                csv_rows.append(row)
    
    # Write to CSV
    if csv_rows:
        import csv
        with open(output_csv, 'w', newline='') as f:
            fieldnames = ['client_ip', 'src_ip', 'src_port', 'dst_ip', 'dst_port', 
                         'min_rtt_ms', 'avg_rtt_ms', 'median_rtt_ms', 'max_rtt_ms', 'num_samples']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"Flow RTT summary written to {output_csv}")
        return output_csv
    else:
        print("No RTT measurements to write")
        return None

def add_rtt_to_switch_logs(switch_datasets, exp_dir, output_dir):
    """
    Add RTT measurements to switch log datasets using per-packet RTT data.
    Filters packets according to requirements:
    - Include background traffic from clients to servers
    - Include request packets (0 length) from client to server
    - Include response packets (>0 length) from servers to client 
    - Exclude ACKs from servers
    - Exclude burst requests from clients to servers
    - Use minimum observed RTT as placeholder for entries with no RTT measurement
    
    Args:
        switch_datasets: List of paths to switch log CSV files
        exp_dir: Path to the experiment directory containing RTT files
        output_dir: Directory to save merged output files
        
    Returns:
        List of paths to the merged dataset files
    """
    print(f"Adding RTT measurements to switch logs")
    
    if not switch_datasets:
        print("Error: No switch datasets provided")
        return []
    
    # Find all RTT CSV files
    rtt_files = []
    for file in os.listdir(exp_dir):
        if (file.startswith('rtt_bg_') or file.startswith('rtt_bursty_')) and file.endswith('.csv'):
            rtt_files.append(os.path.join(exp_dir, file))
    
    if not rtt_files:
        print("Warning: No RTT files found")
        return switch_datasets
    
    print(f"Found {len(rtt_files)} RTT files")
    
    # Load all RTT data into a lookup dictionary for quick matching
    # Key: (src_ip, dst_ip, src_port, dst_port, seq_num, is_response)
    # Value: RTT in milliseconds
    rtt_lookup = {}
    rtt_values = []  # To track all RTT values for finding the minimum
    large_rtt_values = []  # To track RTT values for large packets
    
    # Track client IPs for identification
    client_ips = set()
    bg_client_count = 0
    bursty_client_count = 0
    
    for rtt_file in rtt_files:
        try:
            df = pd.read_csv(rtt_file)
            
            # Extract client IP from filename
            filename = os.path.basename(rtt_file)
            client_ip = None
            is_bursty = False
            
            if filename.startswith('rtt_bg_'):
                client_ip = filename[7:].split('.csv')[0]
                bg_client_count += 1
            elif filename.startswith('rtt_bursty_'):
                client_ip = filename[11:].split('.csv')[0]
                is_bursty = True
                bursty_client_count += 1
            
            if client_ip:
                client_ips.add(client_ip)
            
            # Convert columns to strings to ensure correct matching
            for col in ['src_port', 'dst_port', 'seq_num']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # Make sure we have packet_size column (renamed from payload_len in some files)
            packet_size_col = None
            if 'packet_size' in df.columns:
                packet_size_col = 'packet_size'
            elif 'payload_len' in df.columns:
                packet_size_col = 'payload_len'
            else:
                df['packet_size'] = 0
                packet_size_col = 'packet_size'
                print(f"Warning: No packet size column found in {rtt_file}, using default 0")
            
            # Collect RTT entries with their packet sizes
            for _, row in df.iterrows():
                # Get packet size
                packet_size = int(row[packet_size_col])
                
                # Determine if this is a response packet
                is_response = False
                if client_ip:
                    # For background clients, responses are from server to client
                    if not is_bursty:
                        is_response = (row['src_ip'] != client_ip)
                    # For bursty clients, responses are large packets from server
                    else:
                        is_response = (row['src_ip'] != client_ip and packet_size > 100)
                
                # Create key that includes whether this is a response packet
                key = (
                    row['src_ip'],
                    row['dst_ip'],
                    row['src_port'],
                    row['dst_port'],
                    row['seq_num'],
                    is_response
                )
                rtt_value = float(row['rtt'])
                
                # Add to lookup dictionary
                rtt_lookup[key] = rtt_value
                rtt_values.append(rtt_value)
                
                # Also track RTT values for large packets separately
                if packet_size > 100:
                    large_rtt_values.append(rtt_value)
            
        except Exception as e:
            print(f"Error reading RTT file {rtt_file}: {e}")
            traceback.print_exc()
    
    print(f"Built RTT lookup table with {len(rtt_lookup)} unique entries")
    
    # Calculate minimum RTT value to use as placeholder for small packets
    min_rtt = min(rtt_values) if rtt_values else 0.1  # Default to 0.1ms if no values
    
    # Calculate typical RTT value for large packets (median of large packet RTTs)
    # This will be used when we have a large packet but no direct RTT match
    if large_rtt_values:
        large_packet_rtt = statistics.median(large_rtt_values)
    else:
        # If no large packet RTTs found, use 10x the minimum RTT as a reasonable default
        large_packet_rtt = min_rtt * 10.0 if min_rtt > 0 else 1.0
    
    print(f"Identified {len(client_ips)} client IPs: {bg_client_count} background, {bursty_client_count} bursty")
    
    # Process each switch dataset
    merged_datasets = []
    for i, switch_csv in enumerate(switch_datasets):
        try:
            # Read switch dataset
            switch_df = pd.read_csv(switch_csv)
            
            if switch_df.empty:
                print(f"Warning: Switch dataset {switch_csv} is empty")
                continue
            
            # Convert switch columns to strings to ensure correct matching
            for col in ['src_port', 'dst_port', 'tcp_seq']:
                if col in switch_df.columns:
                    switch_df[col] = switch_df[col].astype(str)
            
            # Initialize RTT column with min_rtt
            switch_df['rtt'] = min_rtt
            
            # Match entries with RTT measurements
            found_count = 0
            total_count = len(switch_df)
            large_packets = 0
            
            for idx, row in switch_df.iterrows():
                # Get packet size
                packet_size = int(row['packet_size']) if 'packet_size' in switch_df.columns else 0
                
                # Determine if this is a response packet
                is_response = False
                if client_ip:
                    # For background clients, responses are from server to client
                    if not is_bursty:
                        is_response = (row['src_ip'] != client_ip)
                    # For bursty clients, responses are large packets from server
                    else:
                        is_response = (row['src_ip'] != client_ip and packet_size > 100)
                
                # Create key that includes whether this is a response packet
                key = (
                    row['src_ip'],
                    row['dst_ip'],
                    row['src_port'],
                    row['dst_port'],
                    row['tcp_seq'],
                    is_response
                )
                
                # Check if we have a direct match
                if key in rtt_lookup:
                    # Found a match - use it
                    switch_df.at[idx, 'rtt'] = rtt_lookup[key]
                    found_count += 1
                else:
                    # No match found, use appropriate default based on packet size
                    if packet_size > 100:
                        # For large packets, use the large packet RTT
                        switch_df.at[idx, 'rtt'] = large_packet_rtt
                        large_packets += 1
                    else:
                        # For small packets, use the minimum RTT
                        switch_df.at[idx, 'rtt'] = min_rtt
            
            # Save merged dataset
            output_file = os.path.join(output_dir, f"s{i+1}_with_rtt.csv")
            switch_df.to_csv(output_file, index=False)
            
            match_rate = (found_count / total_count) * 100 if total_count > 0 else 0
            print(f"Switch {i+1}: Matched {match_rate:.1f}% of packets with RTT measurements")
            print(f"  - {found_count} direct matches")
            print(f"  - {large_packets} large packets with default RTT")
            print(f"  - {total_count - found_count - large_packets} small packets with min RTT")
            
            merged_datasets.append(output_file)
            
        except Exception as e:
            print(f"Error processing switch dataset {switch_csv}: {e}")
            traceback.print_exc()
    
    print(f"Successfully merged RTT data with {len(merged_datasets)} switch datasets")
    return merged_datasets

def create_final_datasets(retr_datasets, rtt_datasets, output_dir):
    """
    Merge datasets with out-of-order flags and RTT measurements to create final datasets.
    
    Args:
        retr_datasets: List of paths to datasets with out-of-order flags
        rtt_datasets: List of paths to datasets with RTT measurements
        output_dir: Directory to save final output files
        
    Returns:
        List of paths to the final dataset files
    """
    print(f"Creating final datasets with out-of-order flags and RTT measurements")
    
    if not retr_datasets:
        print("Error: No datasets with out-of-order flags provided")
        return []
        
    if not rtt_datasets:
        print("Warning: No RTT datasets provided, using out-of-order datasets as final")
        return retr_datasets
    
    # Match datasets by switch number
    final_datasets = []
    for retr_path in retr_datasets:
        try:
            # Extract switch number from filename (s{i}_with_oo.csv)
            retr_filename = os.path.basename(retr_path)
            switch_num = retr_filename.split('_')[0]
            
            # Find corresponding RTT dataset
            rtt_path = None
            for path in rtt_datasets:
                if os.path.basename(path).startswith(switch_num):
                    rtt_path = path
                    break
            
            if rtt_path is None:
                print(f"Warning: No RTT dataset found for {retr_filename}, using out-of-order dataset as final")
                final_path = os.path.join(output_dir, f"{switch_num}_final_dataset.csv")
                import shutil
                shutil.copy(retr_path, final_path)
                final_datasets.append(final_path)
                continue
                
            # Read both datasets
            retr_df = pd.read_csv(retr_path)
            rtt_df = pd.read_csv(rtt_path)
            
            # Ensure 'rtt' column exists in rtt_df
            if 'rtt' not in rtt_df.columns:
                raise ValueError(f"Warning: 'rtt' column not found in {rtt_path}")
            
            # Check if both datasets use the same index
            if len(retr_df) != len(rtt_df):
                print(f"Warning: Dataset sizes don't match for {switch_num} ({len(retr_df)} vs {len(rtt_df)})")
                print("This should be normal. Client and server datasets lengths should not be equal.")
                # Try to merge based on common columns if sizes don't match
                merge_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'tcp_seq']
                if all(col in retr_df.columns for col in merge_cols) and all(col in rtt_df.columns for col in merge_cols):
                    print(f"Attempting to merge on TCP connection parameters")
                    
                    # Convert columns to strings for consistent matching
                    for col in ['src_port', 'dst_port', 'tcp_seq']:
                        if col in retr_df.columns:
                            retr_df[col] = retr_df[col].astype(str)
                        if col in rtt_df.columns:
                            rtt_df[col] = rtt_df[col].astype(str)
                    
                    # Perform left merge to keep all rows from retr_df
                    merged_df = pd.merge(
                        retr_df, 
                        rtt_df[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'tcp_seq', 'rtt']], 
                        on=merge_cols, 
                        how='left'
                    )
                    
                    # Fill NaN values in rtt with 0 (or better default)
                    if 'rtt' in merged_df.columns:
                        # Get minimum non-zero RTT if available
                        min_rtt = merged_df['rtt'].replace(0, float('nan')).min()
                        if pd.isna(min_rtt) or min_rtt == 0:
                            raise ValueError(f"Warning: No valid minimum RTT found in {rtt_path}")
                        
                        # Fill NaN values with minimum RTT
                        merged_df['rtt'] = merged_df['rtt'].fillna(min_rtt)
                        # Also replace zeros with min_rtt
                        merged_df.loc[merged_df['rtt'] == 0, 'rtt'] = min_rtt
                    else:
                        print(f"Error: 'rtt' column not present after merge")
                else:
                    print(f"Error: Cannot merge datasets for {switch_num}, using out-of-order dataset as final")
                    final_path = os.path.join(output_dir, f"{switch_num}_final_dataset.csv")
                    retr_df.to_csv(final_path, index=False)
                    final_datasets.append(final_path)
                    continue
            else:
                # Add RTT column to out-of-order dataset
                merged_df = retr_df.copy()
                if 'rtt' in rtt_df.columns:
                    merged_df['rtt'] = rtt_df['rtt']
                else:
                    raise ValueError(f"Warning: 'rtt' column not found in {rtt_path}")
            
            # Compute rewards for each row
            print(f"Computing rewards for switch {switch_num}")
            merged_df['reward'] = merged_df.apply(compute_reward, axis=1)
            
            # Save final dataset
            final_path = os.path.join(output_dir, f"{switch_num}_final_dataset.csv")
            merged_df.to_csv(final_path, index=False)
            print(f"Created final dataset for switch {switch_num} with {len(merged_df)} rows")
            print(f"RTT stats: min={merged_df['rtt'].min():.4f}ms, avg={merged_df['rtt'].mean():.4f}ms")
            print(f"Reward stats: min={merged_df['reward'].min():.4f}, avg={merged_df['reward'].mean():.4f}")
            final_datasets.append(final_path)
            
        except Exception as e:
            print(f"Error processing datasets for {retr_path}: {e}")
            traceback.print_exc()
    
    print(f"Successfully created {len(final_datasets)} final datasets")
    return final_datasets

def extract_response_rtt_from_pcap(pcap_file, output_csv=None):
    """
    Extract RTT measurements from bursty client PCAP files using a three-tier approach:
    1. For client packets (SYN, query requests): Use tshark's built-in RTT analysis
    2. For server query responses with payload: Calculate RTT as time between request and response
    3. For server ACKs and other packets: Use minimum RTT found across all packets
    
    Args:
        pcap_file: Path to the bursty client PCAP file
        output_csv: Optional path to write the RTT data
        
    Returns:
        Dictionary mapping flow IDs to list of RTT measurements
    """
    # Import needed modules within the function scope
    import csv
    import io
    import subprocess
    
    is_bursty = os.path.basename(pcap_file).startswith('bursty_client_')
    if not is_bursty:
        print(f"Warning: {pcap_file} does not appear to be a bursty client PCAP file")
        return {}
    
    print(f"Processing bursty client PCAP file: {pcap_file}")
    
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
    
    # Run tshark to extract packet information including timestamps, sequence numbers and built-in RTT analysis
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
        '-e', 'tcp.len',
        '-e', 'tcp.flags',
        '-e', 'tcp.analysis.ack_rtt',
        '-o', 'tcp.relative_sequence_numbers: false',
        '-E', 'header=y',
        '-E', 'separator=,'
    ]
    
    try:
        result = subprocess.run(tshark_cmd, capture_output=True, text=True, check=True)
        
        if result.returncode != 0:
            print(f"Error running tshark: {result.stderr}")
            return {}
        
        # Dictionary to track server-client flow information
        # Key: (client_port, server_ip)
        # Value: list of (timestamp, sequence_number, frame_number) tuples for client requests, sorted by timestamp
        client_requests = {}
        
        # Dictionary to store RTT values by flow
        flow_rtts = {}
        
        # List to collect all output rows
        output_rows = []
        
        # First pass: collect all tshark RTT values and track minimum
        valid_rtts = []
        
        # Sort all packets by timestamp first
        all_packets = []
        
        # Parse tshark output for first pass - collect all packets
        csv_reader = csv.reader(io.StringIO(result.stdout))
        headers = next(csv_reader)
        
        for row in csv_reader:
            if len(row) < 10:  # Ensure we have enough fields
                continue
                
            frame_num = row[0]
            timestamp = float(row[1])
            src_ip = row[2]
            dst_ip = row[3]
            src_port = row[4]
            dst_port = row[5]
            seq_num = row[6]  # Raw sequence number
            payload_len = int(row[7]) if row[7] else 0
            flags = row[8]
            tshark_rtt = row[9]
                
            # Extract tshark RTT if available
            if tshark_rtt and tshark_rtt.strip():
                try:
                    rtt = float(tshark_rtt)
                    if rtt > 0:  # Only add positive RTTs
                        valid_rtts.append(rtt)
                except ValueError:
                    pass
                    
            # Add packet to chronological list
            all_packets.append({
                'frame_num': frame_num,
                'timestamp': timestamp,
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'seq_num': seq_num,
                'payload_len': payload_len,
                'flags': flags,
                'tshark_rtt': tshark_rtt
            })
        
        # Calculate minimum RTT (default to 0.1ms if none found)
        min_rtt = min(valid_rtts) if valid_rtts else 0.1
        print(f"Found {len(valid_rtts)} valid tshark RTT measurements")
        
        # Sort packets by timestamp
        all_packets.sort(key=lambda x: x['timestamp'])
        
        # First pass: build client request dictionary for each flow
        for packet in all_packets:
            # Check if this is a client request (from client to server)
            if packet['src_ip'] == client_ip and packet['dst_port'] == '12346':
                flow_key = (packet['src_port'], packet['dst_ip'])
                if flow_key not in client_requests:
                    client_requests[flow_key] = []
                
                # Add request to the flow's list, along with its timestamp and frame number
                client_requests[flow_key].append({
                    'timestamp': packet['timestamp'],
                    'seq_num': packet['seq_num'],
                    'frame_num': packet['frame_num'],
                    'used': False  # Flag to mark if this request has been matched to a response
                })
        
        # Second pass: process all packets and calculate RTTs
        for packet in all_packets:
            frame_num = packet['frame_num']
            timestamp = packet['timestamp']
            src_ip = packet['src_ip']
            dst_ip = packet['dst_ip']
            src_port = packet['src_port']
            dst_port = packet['dst_port']
            seq_num = packet['seq_num']
            payload_len = packet['payload_len']
            flags = packet['flags']
            tshark_rtt = packet['tshark_rtt']
            
            # Generate flow ID for this packet
            flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
            
            # Case 1: Client packets with tshark RTT
            if src_ip == client_ip and tshark_rtt and tshark_rtt.strip():
                try:
                    rtt = float(tshark_rtt)
                except ValueError:
                    # If tshark RTT is invalid, use minimum RTT
                    rtt = min_rtt
            
            # Case 2: Server query responses - find and match with the closest previous request
            elif dst_ip == client_ip and src_port == '12346' and payload_len > 100:
                # This is a server response to client
                flow_key = (dst_port, src_ip)  # Look up client port and server IP
                
                # Try to find a matching request - look for the most recent un-matched request
                if flow_key in client_requests and client_requests[flow_key]:
                    # Get all requests for this flow
                    requests = client_requests[flow_key]
                    
                    # Find the most recent request that happened before this response
                    valid_requests = [req for req in requests 
                                     if (not req['used']) and req['timestamp'] < timestamp]
                    
                    if valid_requests:
                        # Sort by timestamp (most recent first) and get the first one
                        valid_requests.sort(key=lambda x: x['timestamp'], reverse=True)
                        matched_request = valid_requests[0]
                        
                        # Calculate RTT from request to response
                        rtt = (timestamp - matched_request['timestamp'])
                        
                        # Mark this request as used so we don't match it again
                        matched_request['used'] = True
                    else:
                        # No matching request found, use minimum RTT
                        rtt = min_rtt
                else:
                    # No requests for this flow, use minimum RTT
                    rtt = min_rtt
            
            # Case 3: Server ACKs and other packets - use minimum RTT
            else:
                rtt = min_rtt
            
            # Store this RTT value for the flow
            if flow_id not in flow_rtts:
                flow_rtts[flow_id] = []
            flow_rtts[flow_id].append(rtt)
            
            # Add to output rows
            output_rows.append({
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'seq_num': seq_num,  # Use raw sequence number
                'rtt': rtt,
                'packet_size': payload_len
            })
        
        # Write output to CSV
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = [
                'src_ip', 'dst_ip', 'src_port', 'dst_port',
                'seq_num', 'rtt', 'packet_size'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        
        # Count different types for statistics
        client_packets = sum(1 for row in output_rows if row['src_ip'] == client_ip)
        server_packets = sum(1 for row in output_rows if row['dst_ip'] == client_ip)
        server_responses = sum(1 for row in output_rows 
                              if row['dst_ip'] == client_ip 
                              and row['src_port'] == '12346' 
                              and int(row['packet_size']) > 100)
        
        print(f"Processed {len(output_rows)} packets:")
        print(f"  - {client_packets} client packets")
        print(f"  - {server_packets} server packets")
        print(f"  - {server_responses} server response packets")
        print(f"RTT data written to {output_csv}")
        
        return flow_rtts
        
    except Exception as e:
        print(f"Error extracting RTTs from PCAP: {e}")
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
    cleanup_intermediate_files(exp_dir)
    
    # Print summary of created files
    print("\nSummary of created files:")
    print(f" - Final merged datasets: {final_datasets}")
    
    return final_datasets

if __name__ == "__main__":

    
    exp_dir = os.path.join('tmp', '20250401_153049')

    # Process client PCAP files to extract RTT measurements
    # client_rtts = process_client_pcaps_for_rtt(exp_dir)
    
    # Collect switch logs and process them
    topology = LeafSpineTopology(
        num_hosts = 6,
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
