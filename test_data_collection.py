#!/usr/bin/env python3

import os
import time
import sys
import threading
import argparse
import pandas as pd
from topology import DumbbellTopology, LeafSpineTopology, LeafSpineTopologyDeflection
from p4utils.mininetlib.network_API import NetworkAPI
from control_plane import SimpleDeflectionControlPlane, TestControlPlane
from scapy.all import Ether, IP, UDP, Packet, BitField, sendp
import subprocess

from create_rl_dataset import parse_switch_log, merge_by_flow_seq

class BeeHeader(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 28),
        BitField("queue_occ_info", 0, 1),
        BitField("queue_depth", 0, 19) 
    ]

def send_bee_packets(switch_name):
    """Send bee packets to gather queue info from switches"""
    iface = switch_name + '-eth1'
    for port in range(8):
        pkt = (Ether() / 
                IP(src="0.0.0.0", dst="0.0.0.0") /
                UDP(dport=9999) /
                BeeHeader(port_idx_in_reg=port))
        sendp(pkt, iface=iface, verbose=False)

def setup_experiment_directory(exp_id):
    """Create and clean experiment directory"""
    exp_dir = f"tmp/{exp_id}"
    os.makedirs(exp_dir, exist_ok=True)
    os.system(f"sudo rm -rf {exp_dir}/*")
    return exp_dir

def select_clients_and_server(topology, n_clients, client_leaf, server_leaf, n_leaf):
    """Select client and server hosts from the topology"""
    n_hosts = len(topology.net.net.hosts)
    
    # Find hosts connected to the client leaf switch
    clients = []
    for i in range(n_hosts):
        host_idx = i + 1  # Host IDs start from 1
        leaf_idx = (host_idx - 1) % n_leaf + 1
        
        if leaf_idx == client_leaf and len(clients) < n_clients:
            client = topology.net.net.get(f'h{host_idx}')
            clients.append(client)
            print(f"Selected client {client.name} connected to s{client_leaf}")

    # Find a host connected to the server leaf switch
    server = None
    for i in range(n_hosts):
        host_idx = i + 1
        leaf_idx = (host_idx - 1) % n_leaf + 1
        
        if leaf_idx == server_leaf:
            server = topology.net.net.get(f'h{host_idx}')
            print(f"Selected server {server.name} connected to s{server_leaf}")
            break

    if not server:
        raise ValueError("Could not find a server connected to a different leaf")
    if len(clients) < n_clients:
        print(f"Warning: Only found {len(clients)} clients instead of requested {n_clients}")
        
    return clients, server

def start_server_receivers(clients, server, exp_dir):
    """Start receiver processes on the server"""
    receiver_logs = []
    for i, client in enumerate(clients):
        server_receiver_log = f"{exp_dir}/receiver_log_{i+1}.csv"
        receiver_logs.append(server_receiver_log)
        server_log_file = f"{exp_dir}/rx_reord_server_{i+1}.log"
        server.cmd(f'sysctl -w net.core.rmem_max=26214400')  # Increase receive buffer
        server.cmd(f'echo "Starting rx_reord.py on port 520{i+1} at $(date)" > {server_log_file} && '
                  f'python3 rx_reord.py --port 520{i+1} --intf {server.name}-eth0 --log {server_receiver_log} >> {server_log_file} 2>&1 &')
    return receiver_logs

def start_client_senders(clients, server, n_pkts, interval, num_flows, exp_dir):
    """Start traffic generation on client hosts"""
    for i, client in enumerate(clients):
        log_file_str = f'{exp_dir}/tx_reord_client_{i+1}.log'
        intf = f'{client.name}-eth0'        
        print(f"Starting client {client.name} with intf {intf}, IP {client.IP()} sending to server {server.IP()}")
        client_cmd = f"python3 -u tx_reord.py --intf {intf} --src_ip {client.IP()} --dst_ip {server.IP()} --port 520{i+1} \
                --num_packets {n_pkts} --interval {interval} --num_flows {num_flows} > {log_file_str} 2>&1 &"
        client.sendCmd(client_cmd)
    print("All clients started simultaneously")

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
        
    # Save combined receiver log
    receiver_df.to_csv(combined_receiver_log, index=False)
    print(f"Combined {len(receiver_logs)} receiver logs into {combined_receiver_log}")
    
    # Merge switch data with receiver data
    final_dataset = f"{exp_dir}/final_rl_dataset.csv"
    if merge_by_flow_seq(combined_switch_dataset, combined_receiver_log, final_dataset):
        print(f"Created final dataset: {final_dataset}")
    else:
        print("Failed to create final dataset")
        return None
    return final_dataset

def run_mininet(n_pkts, interval, num_flows, n_clients):
    """Run a simple client-server application that sends packets in an incast pattern."""
    # Network parameters
    bw = 1
    delay = 0.01
    exp_id = "0000-deflection"
    queue_rate = 200
    queue_depth = 30
    n_leaf = 4
    n_spine = 3
    p4_program = '/home/ubuntu/p4burst/p4src/sd/sd.p4'
    
    # Create experiment directory
    exp_dir = setup_experiment_directory(exp_id)
    
    # Set up network topology
    n_hosts = n_clients*n_leaf
    topology = LeafSpineTopology(n_hosts, n_leaf, n_spine, bw, delay, p4_program)
    control_plane = SimpleDeflectionControlPlane(topology, queue_rate=queue_rate, queue_depth=queue_depth)
    
    # Start the network
    topology.generate_topology()
    topology.start_network()
    control_plane.generate_control_plane()
    topology.net.program_switches()
    time.sleep(2)
    
    # Send bee packets to all leaf switches
    for i in range(1, n_leaf+1):
        print(f"Sending bee packets to switch s{i}")
        send_bee_packets(f's{i}')
    
    # Select clients and server
    client_leaf = 1  # Clients connected to s1
    server_leaf = 2  # Server connected to s2
    clients, server = select_clients_and_server(topology, n_clients, client_leaf, server_leaf, n_leaf)
    
    # Set up and start server receivers
    receiver_logs = start_server_receivers(clients, server, exp_dir)
    time.sleep(1)
    
    # Start queue logger
    os.system("python3 queue_logger.py &")
    time.sleep(2)
    
    # Start client traffic generators
    print("Starting traffic generation with incast pattern...")
    start_client_senders(clients, server, n_pkts, interval, num_flows, exp_dir)
    
    # Wait for traffic to complete
    wait_time = max(10, n_pkts * interval * 2)
    print(f"Waiting for traffic to complete...")
    time.sleep(wait_time + 30)
    
    # Stop processes
    os.system("killall python3")
    
    # Process collected data
    print("\n--- Creating RL dataset from collected data ---")
    
    # Collect and process switch logs
    switch_datasets = collect_switch_logs(topology, exp_dir)
    
    # Combine all datasets
    final_dataset = combine_datasets(switch_datasets, receiver_logs, exp_dir)
    
    print("--- RL dataset creation complete ---\n")
    
    # Clean shutdown
    print("Stopping network...")
    for node in topology.net.net.hosts + topology.net.net.switches:
        if hasattr(node, 'waiting') and node.waiting:
            node.waitOutput()
    
    topology.net.stopNetwork()
    
    return final_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Mininet with RL data collection.')
    parser.add_argument('--n_pkts', type=int, default=10, help='Number of packets to send')
    parser.add_argument('--n_flows', type=int, default=1, help='Number of flows to generate')
    parser.add_argument('--n_clients', type=int, default=2, help='Number of hosts in the network')
    parser.add_argument('--interval', type=float, default=0.001, help='Interval between packets')
    args = parser.parse_args()

    final_dataset = run_mininet(
        n_pkts=args.n_pkts,
        interval=args.interval, 
        num_flows=args.n_flows, 
        n_clients=args.n_clients
    )
    
    if final_dataset and os.path.exists(final_dataset):
        print(f"Success! Final RL dataset is available at: {final_dataset}")
    else:
        print("Data collection process completed but no final dataset was created.")
