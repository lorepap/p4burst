#!/usr/bin/env python3

from topology import DumbbellTopology, LeafSpineTopology
from p4utils.mininetlib.network_API import NetworkAPI
from control_plane import SimpleDeflectionControlPlane, TestControlPlane
from scapy.all import Ether, IP, UDP, Packet, BitField, sendp
import subprocess
import time
import os
import sys

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
    queue_rate = 10
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
    #topology.net.start_net_cli()

    client_1 = topology.net.net.get('h1')
    client_2 = topology.net.net.get('h3')
    #client_3 = topology.net.net.get('h5')
    # client_4 = topology.net.net.get('h7')
    clients = [client_1, client_2]
    server = topology.net.net.get('h2')

    time.sleep(2)

    # tcp dump
    switch = topology.net.net.get('s2')
    os.makedirs(f"tmp/{exp_id}", exist_ok=True)
    switch.cmd(f"tcpdump -i s1-eth1 > tmp/{exp_id}/s1-eth1_deflection.log &")
    switch.cmd(f"tcpdump -i s1-eth2 > tmp/{exp_id}/s1-eth2_deflection.log &") 
    switch.cmd(f"tcpdump -i s1-eth3 > tmp/{exp_id}/s1-eth3_deflection.log &") 
    switch.cmd(f"tcpdump -i s1-eth4 > tmp/{exp_id}/s1-eth4_deflection.log &") 
    switch.cmd(f"tcpdump -i s1-eth5 > tmp/{exp_id}/s1-eth5_deflection.log &") 
    switch.cmd(f"tcpdump -i s1-eth6 > tmp/{exp_id}/s1-eth6_deflection.log &") 
    
    switch.cmd(f"tcpdump -i s2-eth1 > tmp/{exp_id}/s2-eth1_deflection.log &")
    switch.cmd(f"tcpdump -i s2-eth2 > tmp/{exp_id}/s2-eth2_deflection.log &") 
    switch.cmd(f"tcpdump -i s2-eth3 > tmp/{exp_id}/s2-eth3_deflection.log &") 
    switch.cmd(f"tcpdump -i s2-eth4 > tmp/{exp_id}/s2-eth4_deflection.log &") 
    switch.cmd(f"tcpdump -i s2-eth5 > tmp/{exp_id}/s2-eth5_deflection.log &")


    for i, client in enumerate(clients):
        server.cmd(f"iperf3 -s -p 520{i+1} > tmp/{exp_id}/iperf_server_520{i}.log &")
    time.sleep(1)
    
    # run the queue logger
    os.system("python3 queue_logger.py &")

    # run the client
    procs = []
    for i, client in enumerate(clients):
        print(f"Starting client on {client.name} ({client.IP()})...")
        log_file_str = f'tmp/{exp_id}/iperf_client_{i+1}.log'
        log_file = open(log_file_str, "w")
        client_cmd = "iperf3 -c " + server.IP() + f" -p 520{i+1} -u -b 10M -t 10 -l 200"
        procs.append(client.popen(client_cmd, shell=True, stderr=log_file, stdout=log_file))

    for proc in procs:
        proc.wait()

    # kill the queue logger
    os.system("killall python3")

    """
    # client and server same switch
    # client = topology.net.get('h1')
    # server = topology.net.get('h2')

    os.makedirs("log/deflection_servers", exist_ok=True)
    print(f"Starting server on {server.name} ({server.IP()})...")
    server.cmd(f'python3 -m app --mode server --type deflection_test --host_ip {server.IP()} --exp_id {exp_id} > log/deflection_servers/{server.name}.log 2>&1 &')

    time.sleep(2)

    os.makedirs("log/deflection_clients", exist_ok=True)
    print(f"Starting client on {client.name} ({client.IP()})...")
    client.cmd(f'python3 -m app --mode client --type deflection_test --server_ips {server.IP()} --exp_id {exp_id} > log/deflection_clients/{client.name}.log 2>&1 &')
    """

    topology.net.stopNetwork()

if __name__ == '__main__':
    run_mininet()
