#!/usr/bin/env python3

"""
Test data collection script for RL agent training. Scripts provides code for generating traffic between clients and server,
while collecting packet-level traffic in the switch bmv2.
Traffic is generated with scapy using custom client and server objects.
The server collects packets and logs if they're out of order or not as a feature for the RL dataset.

sender: tx_reord.py
receiver: rx_reord.py

"""

from topology import DumbbellTopology, LeafSpineTopology
from p4utils.mininetlib.network_API import NetworkAPI
from control_plane import SimpleDeflectionControlPlane, TestControlPlane
from scapy.all import Ether, IP, UDP, Packet, BitField, sendp
import subprocess
import time
import os
import sys
import threading

from create_rl_dataset import parse_switch_log, merge_by_flow_seq
import argparse

class BeeHeader(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 31),
        BitField("queue_occ_info", 0, 1)
    ]

def send_bee_packets(switch_name):
    iface = switch_name + '-eth1'
    for port in range(8):
        pkt = (Ether() / 
                IP(src="0.0.0.0", dst="0.0.0.0") /  # Minimal IP header
                UDP(dport=9999) /
                BeeHeader(port_idx_in_reg=port))
        sendp(pkt, iface=iface, verbose=False)


# Function to run Mininet and tests
def run_mininet(n_pkts, interval, num_flows, n_senders):
    """
    Run a simple client-server application that sends one single packet from a client to a server.
    """

    bw = 1
    delay = 0.01
    p4_program = 'sd/sd.p4' 
    exp_id = "0000-deflection"
    queue_rate = 1000
    queue_depth = 30

    # topology = DumbbellTopology(n_hosts, bw, delay, p4_program)
    # Leafspine hosts: 1 server (h2) attached to s2 and N clients attached to same leaf switch s1
    # hosts are placed in a round-robin fashion
    n_hosts = 2*n_senders
    topology = LeafSpineTopology(n_hosts, 2, 2, bw, delay, p4_program)
    control_plane = SimpleDeflectionControlPlane(topology, queue_rate=queue_rate, queue_depth=queue_depth)
    
    os.makedirs(f"tmp/{exp_id}", exist_ok=True)
    # clean old logs
    os.system(f"sudo rm -rf tmp/{exp_id}/*")

    topology.generate_topology()
    topology.start_network()
    control_plane.generate_control_plane()
    topology.net.program_switches()

    time.sleep(2)

    send_bee_packets('s1')
    send_bee_packets('s2')
    # debug
    # topology.net.start_net_cli()
    
    clients = []
    for i in range(n_senders):
        client = topology.net.net.get(f'h{2*i+1}')
        clients.append(client)

    server = topology.net.net.get('h2')

    time.sleep(2)

    #switch = topology.net.net.get('s2')
    os.makedirs(f"tmp/{exp_id}", exist_ok=True)

    # Create separate log file for each client-server pair
    receiver_logs = []
    for i, client in enumerate(clients):
        server_receiver_log = f"tmp/{exp_id}/receiver_log_{i+1}.csv"
        receiver_logs.append(server_receiver_log)
        server_log_file = f"tmp/{exp_id}/rx_reord_server_{i+1}.log"
        server.cmd(f'sysctl -w net.core.rmem_max=26214400')  # Increase receive buffer
        server.cmd(f'echo "Starting rx_reord.py on port 520{i+1} at $(date)" > {server_log_file} && '
                  f'python3 rx_reord.py --port 520{i+1} --intf {server.name}-eth0 --log {server_receiver_log} >> {server_log_file} 2>&1 &')
    time.sleep(1)
    
    # run the queue logger
    os.system("python3 queue_logger.py &")

    # Add synchronization for incast pattern
    time.sleep(2)
    
    print("Starting traffic generation with incast pattern...")

    for i, client in enumerate(clients):
        log_file_str = f'tmp/{exp_id}/tx_reord_client_{i+1}.log'
        intf = f'h{1+2*i}-eth0'        

        client_cmd = f"python3 -u tx_reord.py --intf {intf} --src_ip {client.IP()} --dst_ip {server.IP()} --port 520{i+1} \
                --num_packets {n_pkts} --interval {interval} --num_flows {num_flows} > {log_file_str} 2>&1 &"
        client.sendCmd(client_cmd)
        # All clients should now be running in parallel
        print("All clients started simultaneously")

    # Wait for a reasonable amount of time for traffic to complete
    wait_time = max(10, n_pkts * interval * 2)  # Base wait time on packet count and interval
    print(f"Waiting {wait_time} seconds for traffic to complete...")
    time.sleep(wait_time)

    time.sleep(30)

    # kill the queue logger
    os.system("killall python3")

    # Process collected data to create RL dataset
    print("\n--- Creating RL dataset from collected data ---")
    
    # Path to switch s1 log
    switch_log = "log/p4s.s1.log"
    
    if os.path.exists(switch_log):
        print(f"Processing switch log: {switch_log}")
        # Create intermediate RL dataset from switch logs
        intermediate_dataset = f"tmp/{exp_id}/s1_rl_dataset.csv"
        
        if parse_switch_log(switch_log, intermediate_dataset):
            # First, merge the receiver logs
            combined_receiver_log = f"tmp/{exp_id}/combined_receiver_log.csv"
            
            # Combine the receiver logs
            import pandas as pd
            combined_df = None
            for i, log_file in enumerate(receiver_logs):
                if os.path.exists(log_file):
                    print(f"Reading receiver log: {log_file}")
                    df = pd.read_csv(log_file)
                    if combined_df is None:
                        combined_df = df
                    else:
                        combined_df = pd.concat([combined_df, df], ignore_index=True)
            
            if combined_df is not None and not combined_df.empty:
                # Write the combined log
                combined_df.to_csv(combined_receiver_log, index=False)
                print(f"Combined {len(receiver_logs)} receiver logs into {combined_receiver_log}")
                
                # Merge switch data with combined receiver data
                final_dataset = f"tmp/{exp_id}/final_rl_dataset.csv"
                merge_by_flow_seq(intermediate_dataset, combined_receiver_log, final_dataset)
                print(f"Created final dataset: {final_dataset}")
            else:
                print("Warning: No valid receiver log data found")
        else:
            print(f"Failed to process switch log: {switch_log}")
    else:
        print(f"Warning: Switch log {switch_log} not found")
    
    print("--- RL dataset creation complete ---\n")

    # Ensure clean network shutdown
    print("Stopping network...")
    for node in topology.net.net.hosts + topology.net.net.switches:
        # Make sure no pending commands before stopping
        if hasattr(node, 'waiting') and node.waiting:
            node.waitOutput()

    topology.net.stopNetwork()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Mininet with RL data collection.')
    parser.add_argument('--n_pkts', type=int, default=10, help='Number of packets to send')
    parser.add_argument('--n_flows', type=int, default=1, help='Number of flows to generate')
    parser.add_argument('--n_clients', type=int, default=2, help='Number of hosts in the network')
    parser.add_argument('--interval', type=float, default=0.001, help='Interval between packets')
    args = parser.parse_args()

    n_pkts = args.n_pkts
    interval = args.interval
    num_flows = args.n_flows
    n_clients = args.n_clients
    run_mininet(n_pkts, interval, num_flows, n_clients)

