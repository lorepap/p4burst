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
    """
    
    fw_port_depth = row['fw_port_depth']
    total_queue_depth = row['total_queue_depth']
    reordering_flag = row['retr_flag']
    action = row['action']  # 1=deflect, 0=forward
    
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
    
    # 3. Heavy penalty for reordering
    if reordering_flag == 1:
        reward -= 10  # strong penalty to discourage unnecessary deflection
    
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

def add_retransmission_flags(exp_dir):
    """
    Process server pcap files to identify TCP retransmissions and create CSV files
    with retransmission flags.
    
    Args:
        exp_dir: Path to the experiment directory containing pcap files
        
    Returns:
        List of paths to the generated retransmission data CSV files
    """
    import sys
    sys.path.append('/home/ubuntu/p4burst')
    from retr_flag_processing import add_retransmission_flags as process_retransmissions
    
    print(f"Analyzing TCP retransmissions in experiment: {exp_dir}")
    
    # Call the implementation from retr_flag_processing.py
    output_files = process_retransmissions(exp_dir)
    
    if output_files:
        print(f"Successfully created {len(output_files)} retransmission data files")
        for file in output_files:
            print(f" - {file}")
    else:
        print("No retransmission data files were created")
    
    return output_files

def merge_switch_logs_with_retr_data(switch_datasets, retr_data_files, output_dir):
    """
    Merge switch log datasets with retransmission data files from server pcaps.
    This adds retransmission flags to switch log entries based on matching TCP info.
    
    Args:
        switch_datasets: List of paths to switch log CSV files
        retr_data_files: List of paths to retransmission data CSV files
        output_dir: Directory to save merged output files
        
    Returns:
        List of paths to the merged dataset files
    """
    print(f"Merging switch logs with retransmission data")
    
    if not switch_datasets:
        print("Error: No switch datasets provided")
        return []
        
    if not retr_data_files:
        print("Warning: No retransmission data files provided, using switch logs as-is")
        return switch_datasets
    
    # Load all retransmission data into a single DataFrame
    retr_df = None
    for retr_file in retr_data_files:
        try:
            df = pd.read_csv(retr_file)
            if retr_df is None:
                retr_df = df
            else:
                retr_df = pd.concat([retr_df, df], ignore_index=True)
        except Exception as e:
            print(f"Error reading retransmission file {retr_file}: {e}")
    
    if retr_df is None or retr_df.empty:
        print("Error: Could not load any retransmission data")
        return []
    
    # Create a lookup dictionary for quick retransmission flag retrieval
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
        retr_lookup[key] = int(row['retr_flag'])
    
    print(f"Loaded {len(retr_lookup)} unique TCP packets with retransmission data")
    
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
            
            # Add retr_flag column with default value 0
            switch_df['retr_flag'] = 0
            
            # Match entries with retransmission data
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
                    
                    # Look up retransmission flag
                    if key in retr_lookup:
                        switch_df.at[idx, 'retr_flag'] = retr_lookup[key]
                        found_count += 1
                except (KeyError, ValueError, TypeError) as e:
                    # Skip entries with missing or invalid TCP info
                    continue
            
            # Compute rewards now that we have retransmission flags
            print("Computing rewards using queue depths and retransmission flags...")
            
            # Compute rewards
            switch_df['reward'] = switch_df.apply(compute_reward, axis=1)
            
            # Save merged dataset
            output_file = os.path.join(output_dir, f"s{i+1}_with_retr.csv")
            switch_df.to_csv(output_file, index=False)
            
            print(f"Found {found_count} matching packets with retransmission data")
            print(f"Saved merged dataset to {output_file}")
            
            merged_datasets.append(output_file)
            
        except Exception as e:
            print(f"Error processing switch dataset {switch_csv}: {e}")
            traceback.print_exc()
    
    print(f"Merged {len(merged_datasets)} switch datasets with retransmission data")
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
    print(f"Extracting per-packet RTT measurements from {pcap_file} using tshark")
    
    # If output_csv not specified, create one based on pcap filename
    if output_csv is None:
        output_csv = pcap_file.replace('.pcap', '_rtt.csv')
    
    # Extract client IP from filename
    filename = os.path.basename(pcap_file)
    client_ip = None
    try:
        # Format is typically 'bg_client_10.0.1.1_12345.pcap'
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
                'seq_num', 'ack_num', 'rtt_ms', 'time_relative'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process each row
            packet_count = 0
            rtt_count = 0
            
            for row in csv_reader:
                if len(row) < 8:
                    continue  # Skip incomplete rows
                    
                frame_number, src_ip, dst_ip, src_port, dst_port, seq_num, ack_num, rtt, time_relative = row
                
                # Skip rows without RTT
                if not rtt or rtt == "":
                    continue
                    
                # Only include packets from client to server
                if client_ip and src_ip != client_ip:
                    continue
                
                # Convert RTT from seconds to milliseconds
                try:
                    rtt_ms = float(rtt) * 1000
                except ValueError:
                    continue  # Skip if RTT is not a valid number
                
                # Generate flow ID
                flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                
                # Add to flow RTT dictionary
                if flow_id not in flow_rtts:
                    flow_rtts[flow_id] = []
                flow_rtts[flow_id].append(rtt_ms)
                
                # Write this packet to the output CSV
                writer.writerow({
                    'frame_number': frame_number,
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'seq_num': seq_num,
                    'ack_num': ack_num,
                    'rtt_ms': rtt_ms,
                    'time_relative': time_relative
                })
                
                packet_count += 1
                rtt_count += 1
            
        print(f"Analyzed {packet_count} packets, found {rtt_count} with valid RTT")
        print(f"Per-packet RTT data written to {output_csv}")
        
        # Print sample packet data for debugging
        if packet_count > 0:
            print(f"Sample packet data (first few entries):")
            with open(output_csv) as f:
                sample = list(csv.DictReader(f))[:5]
                for i, packet in enumerate(sample):
                    print(f"Packet {i+1}: src={packet['src_ip']}:{packet['src_port']}, "
                          f"dst={packet['dst_ip']}:{packet['dst_port']}, "
                          f"seq={packet['seq_num']}, rtt={packet['rtt_ms']}ms")
        
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
    print(f"Adding per-packet RTT measurements to switch logs with filtering")
    
    if not switch_datasets:
        print("Error: No switch datasets provided")
        return []
    
    # Find all RTT CSV files
    rtt_files = []
    for file in os.listdir(exp_dir):
        if file.startswith('rtt_') and file.endswith('.csv'):
            rtt_files.append(os.path.join(exp_dir, file))
    
    if not rtt_files:
        print("Warning: No RTT files found")
        return switch_datasets
    
    # Load all RTT data into a lookup dictionary for quick matching
    # Key: (src_ip, dst_ip, src_port, dst_port, seq_num)
    # Value: RTT in milliseconds
    rtt_lookup = {}
    rtt_values = []  # To track all RTT values for finding the minimum
    
    # Track client IPs for identification
    client_ips = set()
    
    for rtt_file in rtt_files:
        try:
            print(f"Loading RTT data from {rtt_file}")
            df = pd.read_csv(rtt_file)
            
            # Extract client IP from filename
            filename = os.path.basename(rtt_file)
            if filename.startswith('rtt_'):
                client_ip = filename[4:].split('.csv')[0]
                client_ips.add(client_ip)
                print(f"Identified client IP: {client_ip}")
            
            # Convert columns to strings to ensure correct matching
            for col in ['src_port', 'dst_port', 'seq_num']:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            
            # Create lookup dictionary and collect RTT values
            for _, row in df.iterrows():
                key = (
                    row['src_ip'],
                    row['dst_ip'],
                    row['src_port'],
                    row['dst_port'],
                    row['seq_num']
                )
                rtt_value = float(row['rtt_ms'])
                rtt_lookup[key] = rtt_value
                rtt_values.append(rtt_value)
            
            print(f"Loaded {len(df)} RTT measurements from {rtt_file}")
        except Exception as e:
            print(f"Error reading RTT file {rtt_file}: {e}")
            traceback.print_exc()
    
    # Calculate minimum RTT value to use as placeholder
    min_rtt = min(rtt_values) if rtt_values else 0.1  # Default to 0.1ms if no values
    print(f"Total RTT lookup entries: {len(rtt_lookup)}")
    print(f"Identified client IPs: {client_ips}")
    print(f"Using minimum RTT value of {min_rtt:.4f}ms as placeholder for unmeasured packets")
    
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
            
            print(f"Total switch entries: {len(switch_df)} / RTT lookup entries: {len(rtt_lookup)}")
            
            # Convert switch columns to strings to ensure correct matching
            for col in ['src_port', 'dst_port', 'tcp_seq']:
                if col in switch_df.columns:
                    switch_df[col] = switch_df[col].astype(str)
            
            # Initialize RTT column with min_rtt
            switch_df['rtt_ms'] = min_rtt
            
            # Match entries with RTT measurements
            found_count = 0
            total_count = len(switch_df)
            
            for idx, row in switch_df.iterrows():
                key = (
                    row['src_ip'],
                    row['dst_ip'], 
                    row['src_port'],
                    row['dst_port'],
                    row['tcp_seq']
                )
                
                if key in rtt_lookup:
                    switch_df.at[idx, 'rtt_ms'] = rtt_lookup[key]
                    found_count += 1
            
            # Save merged dataset
            output_file = os.path.join(output_dir, f"s{i+1}_with_rtt.csv")
            switch_df.to_csv(output_file, index=False)
            
            match_rate = (found_count / total_count) * 100 if total_count > 0 else 0
            print(f"Found {found_count}/{total_count} ({match_rate:.2f}%) matching packets with RTT measurements")
            print(f"Using min RTT ({min_rtt:.4f}ms) for {total_count - found_count} packets")
            print(f"Saved merged dataset with {len(switch_df)} packets to {output_file}")
            
            if found_count == 0 and total_count > 0:
                print("WARNING: No RTT matches found! Common causes:")
                print("1. TCP sequence numbers in different formats")
                print("2. Port numbers in different formats (string vs int)")
                print("3. Different IP formats (e.g., '10.0.0.1' vs '10.0.0.001')")
                print("Dumping first few entries from switch dataset for debugging:")
                for col in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'tcp_seq']:
                    if col in switch_df.columns:
                        print(f"{col} (first 5): {switch_df[col].head().tolist()}")
                
                print("Sample keys from RTT lookup:")
                sample_keys = list(rtt_lookup.keys())[:5]
                for key in sample_keys:
                    print(f"RTT key: {key} -> {rtt_lookup[key]}")
            
            merged_datasets.append(output_file)
            
        except Exception as e:
            print(f"Error processing switch dataset {switch_csv}: {e}")
            traceback.print_exc()
    print(f"Merged {len(merged_datasets)} switch datasets with RTT measurements")
    return merged_datasets

def create_final_datasets(retr_datasets, rtt_datasets, output_dir):
    """
    Merge datasets with retransmission flags and RTT measurements to create final datasets.
    
    Args:
        retr_datasets: List of paths to datasets with retransmission flags
        rtt_datasets: List of paths to datasets with RTT measurements
        output_dir: Directory to save final output files
        
    Returns:
        List of paths to the final dataset files
    """
    print(f"Creating final datasets with both retransmission flags and RTT measurements")
    
    if not retr_datasets:
        print("Error: No datasets with retransmission flags provided")
        return []
        
    if not rtt_datasets:
        print("Warning: No RTT datasets provided, using retransmission datasets as final")
        return retr_datasets
    
    # Match datasets by switch number
    final_datasets = []
    for retr_path in retr_datasets:
        try:
            # Extract switch number from filename (s{i}_with_retr.csv)
            retr_filename = os.path.basename(retr_path)
            switch_num = retr_filename.split('_')[0]
            
            # Find corresponding RTT dataset
            rtt_path = None
            for path in rtt_datasets:
                if os.path.basename(path).startswith(switch_num):
                    rtt_path = path
                    break
            
            if rtt_path is None:
                print(f"Warning: No RTT dataset found for {retr_filename}, using retransmission dataset as final")
                final_path = os.path.join(output_dir, f"{switch_num}_final_dataset.csv")
                import shutil
                shutil.copy(retr_path, final_path)
                final_datasets.append(final_path)
                continue
                
            # Read both datasets
            retr_df = pd.read_csv(retr_path)
            rtt_df = pd.read_csv(rtt_path)
            
            # Ensure 'rtt_ms' column exists in rtt_df
            if 'rtt_ms' not in rtt_df.columns:
                print(f"Warning: 'rtt_ms' column not found in {rtt_path}, adding default value 0")
                rtt_df['rtt_ms'] = 0.0
            
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
                        rtt_df[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'tcp_seq', 'rtt_ms']], 
                        on=merge_cols, 
                        how='left'
                    )
                    
                    # Fill NaN values in rtt_ms with 0 (or better default)
                    if 'rtt_ms' in merged_df.columns:
                        # Get minimum non-zero RTT if available
                        min_rtt = merged_df['rtt_ms'].replace(0, float('nan')).min()
                        if pd.isna(min_rtt) or min_rtt == 0:
                            min_rtt = 0.1  # default if no valid minimum
                        
                        # Fill NaN values with minimum RTT
                        merged_df['rtt_ms'] = merged_df['rtt_ms'].fillna(min_rtt)
                        # Also replace zeros with min_rtt
                        merged_df.loc[merged_df['rtt_ms'] == 0, 'rtt_ms'] = min_rtt
                    else:
                        print(f"Error: 'rtt_ms' column not present after merge")
                else:
                    print(f"Error: Cannot merge datasets for {switch_num}, using retransmission dataset as final")
                    final_path = os.path.join(output_dir, f"{switch_num}_final_dataset.csv")
                    retr_df.to_csv(final_path, index=False)
                    final_datasets.append(final_path)
                    continue
            else:
                # Add RTT column to retransmission dataset
                merged_df = retr_df.copy()
                if 'rtt_ms' in rtt_df.columns:
                    merged_df['rtt_ms'] = rtt_df['rtt_ms']
                else:
                    print(f"Warning: 'rtt_ms' column not found in {rtt_path} - using default")
                    # Get a reasonable default RTT if available elsewhere
                    min_rtt = 0.1  # default
                    for path in rtt_datasets:
                        try:
                            temp_df = pd.read_csv(path)
                            if 'rtt_ms' in temp_df.columns:
                                non_zero_min = temp_df['rtt_ms'].replace(0, float('nan')).min()
                                if not pd.isna(non_zero_min) and non_zero_min > 0:
                                    min_rtt = non_zero_min
                                    break
                        except:
                            pass
                    merged_df['rtt_ms'] = min_rtt
            
            # Save final dataset
            final_path = os.path.join(output_dir, f"{switch_num}_final_dataset.csv")
            merged_df.to_csv(final_path, index=False)
            print(f"Created final dataset at {final_path} with {len(merged_df)} rows")
            print(f"RTT stats: min={merged_df['rtt_ms'].min():.4f}ms, avg={merged_df['rtt_ms'].mean():.4f}ms")
            final_datasets.append(final_path)
            
        except Exception as e:
            print(f"Error processing datasets for {retr_path}: {e}")
            traceback.print_exc()
    
    print(f"Created {len(final_datasets)} final datasets")
    return final_datasets

def process_and_merge_all_data(topology, exp_dir):
    """
    Complete workflow to process all data:
    1. Process retransmission flags
    2. Collect switch logs
    3. Merge switch logs with retransmission flags
    4. Compute RTT from client PCAPs (per-packet)
    5. Merge switch logs with per-packet RTT measurements
    6. Create final datasets with both retransmission flags and RTT
    
    This is the main function to be called from collection_runner.py.
    
    Args:
        topology: The network topology object
        exp_dir: Path to the experiment directory
        
    Returns:
        List of paths to the final dataset files
    """
    print(f"Processing all data for experiment in {exp_dir}")
    
    # Step 1: Process retransmission flags
    logs_w_retr_flag = add_retransmission_flags(exp_dir)
    if logs_w_retr_flag:
        print("Successfully created retransmission data files")
    else:
        print("Warning: No retransmission data files created")
    
    # Step 2: Collect switch logs
    switch_datasets = collect_switch_logs(topology, exp_dir)
    
    if not switch_datasets:
        print("Warning: No switch datasets found, cannot proceed")
        return []
    
    # Step 3: Merge switch logs with retransmission flags
    retr_datasets = merge_switch_logs_with_retr_data(switch_datasets, logs_w_retr_flag, exp_dir)
    if not retr_datasets:
        print("Warning: Failed to merge switch logs with retransmission flags")
        retr_datasets = switch_datasets  # Use original switch datasets as fallback
    
    # Step 4: Extract per-packet RTT from client PCAPs
    print("Processing client PCAP files for per-packet RTT data")
    # Find all client PCAP files
    pcap_files = []
    for file in os.listdir(exp_dir):
        if file.startswith('bg_client_') and file.endswith('.pcap'):
            pcap_files.append(os.path.join(exp_dir, file))
    
    if pcap_files:
        # Process each PCAP file to extract per-packet RTT
        for pcap_file in pcap_files:
            try:
                # Extract client IP from filename
                filename = os.path.basename(pcap_file)
                parts = filename.split('_')
                client_ip = parts[2] if len(parts) >= 3 else None
                if client_ip and client_ip.endswith('.pcap'):
                    client_ip = client_ip[:-5]
                
                print(f"Processing PCAP file for client {client_ip}")
                output_csv = os.path.join(exp_dir, f"rtt_{client_ip}.csv")
                extract_rtt_using_tshark(pcap_file, output_csv)
            except Exception as e:
                print(f"Error processing PCAP file {pcap_file}: {e}")
                traceback.print_exc()
    else:
        print("Warning: No client PCAP files found")
    
    # Step 5: Merge switch logs with per-packet RTT measurements
    rtt_datasets = add_rtt_to_switch_logs(retr_datasets, exp_dir, exp_dir)
    if not rtt_datasets:
        print("Warning: Failed to merge switch logs with RTT measurements")
        rtt_datasets = retr_datasets
    
    # Step 6: Create final datasets
    print("Creating final datasets")
    final_datasets = create_final_datasets(retr_datasets, rtt_datasets, exp_dir)
    
    # Print summary of created files
    print("\nSummary of created files:")
    print(f" - Switch datasets: {switch_datasets}")
    print(f" - Datasets with retransmission flags: {retr_datasets}")
    print(f" - Datasets with RTT measurements: {rtt_datasets}")
    print(f" - Final merged datasets: {final_datasets}")
    
    return final_datasets

if __name__ == "__main__":
    exp_dir = os.path.join('tmp', '20250331_211630')

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
