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
def run_mininet():
    """
    Run a simple client-server application that sends one single packet from a client to a server.
    """

    n_hosts = 5
    bw = 10
    delay = 0
    p4_program = 'sd/sd.p4' 
    exp_id = "0000-deflection"
    queue_rate = 1000
    queue_depth = 30

    # topology = DumbbellTopology(n_hosts, bw, delay, p4_program)
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

    client_1 = topology.net.net.get('h1')
    client_2 = topology.net.net.get('h3')
    #client_3 = topology.net.net.get('h5')
    # client_4 = topology.net.net.get('h7')
    clients = [client_1]
    server = topology.net.net.get('h2')

    time.sleep(2)

    # tcp dump
    switch = topology.net.net.get('s2')
    os.makedirs(f"tmp/{exp_id}", exist_ok=True)

    receiver_log = f"tmp/{exp_id}/receiver_log.csv"
    for i, client in enumerate(clients):
        # run server + sniffer in rx_reord
        server_log_file = f"tmp/{exp_id}/rx_reord_server_{i+1}.log"
        server.cmd(f'python3 rx_reord.py --port 520{i+1} --intf {server.name}-eth0 --log {receiver_log} > {server_log_file} 2>&1 &')
    time.sleep(1)
    
    # run the queue logger
    os.system("python3 queue_logger.py &")

    # run the clients
    n_pkts = 10
    interval = 0.001
    procs = []
    for i, client in enumerate(clients):
        print(f"Starting client on {client.name} ({client.IP()})...")
        log_file_str = f'tmp/{exp_id}/tx_reord_client_{i+1}.log'
        log_file = open(log_file_str, "w")
        intf = f'h{1+2*i}-eth0' # h1, h3, h5, h7, .. can we use client.defaultIntf()?
        client_cmd = f"python3 tx_reord.py --intf {intf} --src_ip {client.IP()} --dst_ip {server.IP()} --port 520{i+1} \
                --num_packets {n_pkts} --interval {interval}"
        #procs.append(client.popen(client_cmd, shell=True, stderr=log_file, stdout=log_file))
        client.cmd(f'{client_cmd} > {log_file_str} 2>&1 &')

    # for proc in procs:
    #     proc.wait()

    time.sleep(10)

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
        parse_switch_log(switch_log, intermediate_dataset)

        # Merge switch data with receiver data
        final_dataset = f"tmp/{exp_id}/final_rl_dataset.csv"
        merge_by_flow_seq(intermediate_dataset, receiver_log, final_dataset)
        print(f"Created final dataset: {final_dataset}")
    else:
        print(f"Warning: Switch log {switch_log} not found")
    
    print("--- RL dataset creation complete ---\n")

    topology.net.stopNetwork()

if __name__ == '__main__':
    run_mininet()
