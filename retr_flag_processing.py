import sys
import pyshark
import os
import pandas as pd
from datetime import datetime
import glob

def analyze_server_pcap(pcap_file, output_csv):
    """
    Analyze a single server pcap file to detect out-of-order packets.
    
    Args:
        pcap_file: Path to the server pcap file
        output_csv: Path to save the CSV output
        
    Returns:
        Tuple of (total_packets, out_of_order_packets)
    """
    print(f"Processing PCAP file: {pcap_file}")
    
    try:
        # Open the capture file
        capture = pyshark.FileCapture(
            pcap_file,
            tshark_path='/usr/bin/tshark'
        )
        
        # Dictionary to track flow state
        # Key: (src_ip, src_port, dst_ip, dst_port)
        # Value: dict containing:
        #   - expected_seq: next expected sequence number
        #   - max_seq: highest sequence number seen
        flow_state = {}
        
        # List to store packet details with out-of-order flags
        packet_entries = []
        
        print("Processing packets to identify out-of-order packets...")
        for pkt in capture:
            try:
                # Skip non-TCP packets
                if not hasattr(pkt, 'tcp') or not hasattr(pkt, 'ip'):
                    continue
                    
                # Extract TCP and IP information
                tcp = pkt.tcp
                ip = pkt.ip
                
                # Create flow key
                flow_key = (ip.src, int(tcp.srcport), ip.dst, int(tcp.dstport))
                
                # Extract sequence number using raw (absolute) value
                try:
                    seq_num = int(tcp.seq_raw)
                except AttributeError:
                    raise ValueError("No sequence number found in packet")
                
                # Try to get payload length
                try:
                    payload_len = int(tcp.len)
                except AttributeError:
                    raise ValueError("No payload length found in packet")
                
                # Initialize flow state if not seen before
                if flow_key not in flow_state:
                    flow_state[flow_key] = {
                        'expected_seq': seq_num + payload_len,
                        'max_seq': seq_num + payload_len
                    }
                    is_out_of_order = 0
                else:
                    state = flow_state[flow_key]
                    
                    # Check if this is an out-of-order packet
                    is_out_of_order = 0
                    
                    # A packet is out-of-order if:
                    # 1. Its sequence number is less than the expected sequence number (if greater, it's in order, perhaps the expected packet has been dropped)
                    # 2. It's not the first packet of the connection
                    if seq_num < state['expected_seq'] and state['max_seq'] > seq_num:
                        is_out_of_order = 1
                    
                    # Update flow state
                    state['max_seq'] = max(state['max_seq'], seq_num + payload_len)
                    
                    # Update expected sequence number if this packet is in order
                    if seq_num == state['expected_seq']:
                        state['expected_seq'] = seq_num + payload_len
                
                # Get timestamp in a readable format
                try:
                    timestamp = pkt.sniff_time
                except AttributeError:
                    timestamp = datetime.now()
                
                # Add entry with out-of-order flag
                packet_entries.append({
                    'timestamp': timestamp,
                    'src_ip': ip.src,
                    'dst_ip': ip.dst,
                    'src_port': int(tcp.srcport),
                    'dst_port': int(tcp.dstport),
                    'tcp_seq': seq_num,
                    'payload_length': payload_len,
                    'out_of_order_flag': is_out_of_order
                })
                
            except AttributeError:
                # Skip packets that don't have the required fields
                continue
        
        capture.close()
        
        # Create DataFrame from collected entries
        df = pd.DataFrame(packet_entries)
        
        if df.empty:
            print("No packets found matching the filter criteria")
            return 0, 0
        
        # Save the data to CSV
        df.to_csv(output_csv, index=False)
        
        total_packets = len(df)
        out_of_order_packets = df['out_of_order_flag'].sum()
        
        return total_packets, out_of_order_packets
        
    except Exception as e:
        print(f"Error processing packet capture {pcap_file}: {e}")
        return 0, 0

def add_out_of_order_flags(exp_dir):
    """
    Find all server pcap files in the experiment directory, analyze them for
    out-of-order packets, and create corresponding CSV files.
    
    Args:
        exp_dir: Experiment directory containing pcap files
        
    Returns:
        List of created CSV files
    """
    print(f"Analyzing out-of-order packets in experiment: {exp_dir}")
    
    # Find all server pcap files
    server_pcap_pattern = os.path.join(exp_dir, 'bg_server_*.pcap')
    server_pcap_files = glob.glob(server_pcap_pattern)
    
    if not server_pcap_files:
        print(f"Error: No server pcap files found in {exp_dir}")
        return []
    
    print(f"Found {len(server_pcap_files)} server pcap files")
    
    created_files = []
    total_stats = {'packets': 0, 'out_of_order': 0}
    
    # Process each server pcap file
    for pcap_file in server_pcap_files:
        # Extract server info from filename
        filename = os.path.basename(pcap_file)
        server_info = filename.replace('background_server_', '').replace('.pcap', '')
        
        # Create output CSV name
        output_csv = os.path.join(exp_dir, f'out_of_order_data_{server_info}.csv')
        
        # Process the pcap file
        packets, out_of_order = analyze_server_pcap(pcap_file, output_csv)
        
        if packets > 0:
            print(f"Server {server_info}: Processed {packets} packets, found {out_of_order} out-of-order packets")
            created_files.append(output_csv)
            total_stats['packets'] += packets
            total_stats['out_of_order'] += out_of_order
    
    # Print summary
    print("\nOut-of-Order Analysis Summary:")
    print(f"Total packets processed: {total_stats['packets']}")
    print(f"Total out-of-order packets detected: {total_stats['out_of_order']}")
    print(f"Created {len(created_files)} out-of-order data files")
    
    return created_files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pcap_analysis.py <experiment_dir>")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    
    output_files = add_out_of_order_flags(exp_dir)
    
    if output_files:
        print(f"Successfully created out-of-order data files:")
        for file in output_files:
            print(f" - {file}")
    else:
        print("No out-of-order data files were created")