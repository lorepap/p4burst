import sys
import pyshark
import os
import pandas as pd
from datetime import datetime
import glob

def analyze_server_pcap(pcap_file, output_csv):
    """
    Analyze a single server pcap file to detect retransmissions.
    
    Args:
        pcap_file: Path to the server pcap file
        output_csv: Path to save the CSV output
        
    Returns:
        Tuple of (total_packets, retransmissions)
    """
    print(f"Processing PCAP file: {pcap_file}")
    
    try:
        # Open the capture file
        capture = pyshark.FileCapture(
            pcap_file,
            tshark_path='/usr/bin/tshark'
        )
        
        # Dictionary to track which sequence numbers have been seen for each flow
        # Key: (src_ip, src_port, dst_ip, dst_port)
        # Value: dictionary of sequence numbers to sets of payload lengths
        flow_seq_nums = {}
        
        # List to store packet details with retransmission flags
        packet_entries = []
        
        print("Processing packets to identify retransmissions...")
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
                    # First try to get the raw sequence number
                    seq_num = int(tcp.seq_raw)
                except AttributeError:
                    # Fall back to regular sequence - might be relative
                    seq_num = int(tcp.seq)
                    
                    # If this is TCP flags suggest this is a SYN packet, it's likely the ISN
                    is_syn = hasattr(tcp, 'flags_syn') and tcp.flags_syn == '1'
                    if not is_syn and hasattr(tcp, 'analysis_initial_rtt'):
                        # This appears to be using relative sequence numbers
                        # Add warning that sequence numbers might not match
                        print("Warning: Using relative sequence numbers - may not match P4 logs")
                
                # Try to get payload length
                try:
                    payload_len = int(tcp.len)
                except AttributeError:
                    payload_len = 0
                
                # Check if this flow has been seen before
                if flow_key not in flow_seq_nums:
                    flow_seq_nums[flow_key] = {}
                    is_retransmission = 0
                else:
                    # More accurate retransmission detection
                    is_retransmission = 0
                    if seq_num in flow_seq_nums[flow_key]:
                        # Check if we've seen this sequence number with a payload before
                        previous_payloads = flow_seq_nums[flow_key][seq_num]
                        # If we've seen this exact sequence number with a payload, it's a retransmission
                        if payload_len > 0 and payload_len in previous_payloads:
                            is_retransmission = 1

                # Store sequence number with its payload length
                if seq_num not in flow_seq_nums[flow_key]:
                    flow_seq_nums[flow_key][seq_num] = set()
                flow_seq_nums[flow_key][seq_num].add(payload_len)
                
                # Get timestamp in a readable format
                try:
                    timestamp = pkt.sniff_time
                except AttributeError:
                    timestamp = datetime.now()
                
                # Add entry with retransmission flag
                packet_entries.append({
                    'timestamp': timestamp,
                    'src_ip': ip.src,
                    'dst_ip': ip.dst,
                    'src_port': int(tcp.srcport),
                    'dst_port': int(tcp.dstport),
                    'tcp_seq': seq_num,
                    'payload_length': payload_len,
                    'retr_flag': is_retransmission
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
        retransmissions = df['retr_flag'].sum()
        
        return total_packets, retransmissions
        
    except Exception as e:
        print(f"Error processing packet capture {pcap_file}: {e}")
        return 0, 0

def add_retransmission_flags(exp_dir):
    """
    Find all server pcap files in the experiment directory, analyze them for
    retransmissions, and create corresponding CSV files.
    
    Args:
        exp_dir: Experiment directory containing pcap files
        
    Returns:
        List of created CSV files
    """
    print(f"Analyzing retransmissions in experiment: {exp_dir}")
    
    # Find all server pcap files
    server_pcap_pattern = os.path.join(exp_dir, 'bg_server_*.pcap')
    server_pcap_files = glob.glob(server_pcap_pattern)
    
    if not server_pcap_files:
        print(f"Error: No server pcap files found in {exp_dir}")
        return []
    
    print(f"Found {len(server_pcap_files)} server pcap files")
    
    created_files = []
    total_stats = {'packets': 0, 'retransmissions': 0}
    
    # Process each server pcap file
    for pcap_file in server_pcap_files:
        # Extract server info from filename
        filename = os.path.basename(pcap_file)
        server_info = filename.replace('background_server_', '').replace('.pcap', '')
        
        # Create output CSV name
        output_csv = os.path.join(exp_dir, f'retransmission_data_{server_info}.csv')
        
        # Process the pcap file
        packets, retransmissions = analyze_server_pcap(pcap_file, output_csv)
        
        if packets > 0:
            print(f"Server {server_info}: Processed {packets} packets, found {retransmissions} retransmissions")
            created_files.append(output_csv)
            total_stats['packets'] += packets
            total_stats['retransmissions'] += retransmissions
    
    # Print summary
    print("\nRetransmission Analysis Summary:")
    print(f"Total packets processed: {total_stats['packets']}")
    print(f"Total retransmissions detected: {total_stats['retransmissions']}")
    print(f"Created {len(created_files)} retransmission data files")
    
    return created_files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pcap_analysis.py <experiment_dir>")
        sys.exit(1)
    
    exp_dir = sys.argv[1]
    
    output_files = add_retransmission_flags(exp_dir)
    
    if output_files:
        print(f"Successfully created retransmission data files:")
        for file in output_files:
            print(f" - {file}")
    else:
        print("No retransmission data files were created")