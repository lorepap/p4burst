#!/usr/bin/env python3

import argparse
import subprocess
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_link_utilization(pcap_file, output_dir, client_name, bw_mbps):
    """
    Calculate link utilization for a given client from a pcap file
    
    Args:
        pcap_file: Path to the pcap file
        output_dir: Directory to store output files
        client_name: Name of the client
        bw_mbps: Link bandwidth in Mbps
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(pcap_file):
        print(f"Error: pcap file {pcap_file} does not exist.")
        return False
    
    # Make sure output directories exist
    util_dir = os.path.join(output_dir, "utilization")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(util_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Extract the client IP - update the extraction to handle the format bg_client_IP_PORT properly
    try:
        # Extract IP address from the client name (assuming format like bg_client_10.0.2.2_12345)
        parts = client_name.split('_')
        if len(parts) >= 3:
            client_ip = parts[-2]  # Get the second-to-last part, which should be the IP
            print(f"Extracted client IP: {client_ip}")
        else:
            print(f"Could not extract client IP from {client_name}, using default filter")
            client_ip = None
    except Exception as e:
        print(f"Error extracting client IP: {e}")
        client_ip = None
    
    # Convert Mbps to Bits per second (bps) for calculations (fix the conversion)
    bw_bps = bw_mbps * 1000 * 1000  # Mbps to bps
    print(f"Link bandwidth: {bw_mbps} Mbps = {bw_bps} bps")
    
    # Output packet info file
    packet_csv = os.path.join(output_dir, f"{client_name}_packets.csv")
    
    # Use tshark to extract packet information to CSV
    tshark_cmd = [
        "tshark", "-r", pcap_file,
    ]
    
    # Only add IP filter if we could extract a valid IP
    if client_ip and '.' in client_ip:
        tshark_cmd.extend(["-Y", f"ip.src == {client_ip} or ip.dst == {client_ip}"])
    
    tshark_cmd.extend([
        "-T", "fields",
        "-e", "frame.time_epoch",
        "-e", "frame.len",
        "-E", "header=y",
        "-E", "separator=,",
    ])
    
    try:
        with open(packet_csv, 'w') as f:
            subprocess.run(tshark_cmd, stdout=f, check=True)
        print(f"Extracted packet data to {packet_csv}")
    except subprocess.CalledProcessError as e:
        print(f"Error running tshark: {e}")
        return False
    
    # Read packet data into dataframe
    try:
        df = pd.read_csv(packet_csv)
        df.columns = ['timestamp', 'length']
    except Exception as e:
        print(f"Error reading packet data: {e}")
        return False
    
    if len(df) == 0:
        print(f"No packets found for client {client_name} in pcap file.")
        return False
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Normalize timestamps to start at 0
    start_time = df['timestamp'].min()
    df['timestamp'] = df['timestamp'] - start_time
    
    # Create time bins for throughput calculation (100ms bins)
    bin_size = 0.1  # seconds
    max_time = df['timestamp'].max()
    bins = np.arange(0, max_time + bin_size, bin_size)
    
    # Assign packets to bins
    df['time_bin'] = pd.cut(df['timestamp'], bins, right=False, labels=bins[:-1])
    
    # Calculate bytes per bin
    throughput = df.groupby('time_bin')['length'].sum().reset_index()
    throughput['throughput_bps'] = throughput['length'] * 8 / bin_size  # Convert to bits per second
    throughput['throughput_mbps'] = throughput['throughput_bps'] / (1000 * 1000)  # Convert to Mbps
    throughput['utilization_percent'] = (throughput['throughput_bps'] / bw_bps) * 100
    
    # Fill in missing bins with zero throughput
    throughput_full = pd.DataFrame({'time_bin': bins[:-1]})
    throughput_full = throughput_full.merge(throughput, on='time_bin', how='left').fillna(0)
    
    # Add time in seconds column for easier plotting
    throughput_full['time_sec'] = throughput_full['time_bin']
    
    # Save utilization data to CSV
    csv_output = os.path.join(util_dir, f"{client_name}_utilization.csv")
    throughput_full[['time_sec', 'throughput_mbps', 'utilization_percent']].to_csv(csv_output, index=False)
    print(f"Saved utilization data to {csv_output}")
    
    # Create time series plot of throughput
    plt.figure(figsize=(12, 6))
    plt.plot(throughput_full['time_sec'], throughput_full['throughput_mbps'])
    plt.title(f'Throughput Over Time - {client_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (Mbps)')
    plt.grid(True)
    plt.tight_layout()
    
    throughput_plot = os.path.join(plots_dir, f"{client_name}_throughput.png")
    plt.savefig(throughput_plot)
    plt.close()
    print(f"Saved throughput plot to {throughput_plot}")
    
    # Create time series plot of utilization
    plt.figure(figsize=(12, 6))
    plt.plot(throughput_full['time_sec'], throughput_full['utilization_percent'])
    plt.title(f'Link Utilization Over Time - {client_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Utilization (%)')
    plt.axhline(y=100, color='r', linestyle='--', label='Link Capacity')
    plt.grid(True)
    plt.tight_layout()
    
    utilization_plot = os.path.join(plots_dir, f"{client_name}_utilization.png")
    plt.savefig(utilization_plot)
    plt.close()
    print(f"Saved utilization plot to {utilization_plot}")
    
    # Print summary statistics
    avg_throughput = throughput_full['throughput_mbps'].mean()
    peak_throughput = throughput_full['throughput_mbps'].max()
    avg_utilization = throughput_full['utilization_percent'].mean()
    peak_utilization = throughput_full['utilization_percent'].max()
    
    print(f"\nLink utilization summary for {client_name}:")
    print(f"  Average throughput: {avg_throughput:.2f} Mbps")
    print(f"  Peak throughput: {peak_throughput:.2f} Mbps")
    print(f"  Average link utilization: {avg_utilization:.2f}%")
    print(f"  Peak link utilization: {peak_utilization:.2f}%")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Calculate link utilization from pcap file')
    parser.add_argument('pcap_file', help='Path to the pcap file')
    parser.add_argument('output_dir', help='Directory to store output files')
    parser.add_argument('client_name', help='Name of the client')
    parser.add_argument('--bw', type=float, default=100, help='Link bandwidth in Mbps (default: 100)')
    
    args = parser.parse_args()
    calculate_link_utilization(args.pcap_file, args.output_dir, args.client_name, args.bw)

if __name__ == '__main__':
    main() 