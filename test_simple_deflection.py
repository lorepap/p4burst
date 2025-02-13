#!/usr/bin/env python3

from topology import DumbbellTopology, LeafSpineTopology
from p4utils.mininetlib.network_API import NetworkAPI
from control_plane import SimpleDeflectionControlPlane, TestControlPlane
from scapy.all import Ether, IP, UDP, Packet, BitField, sendp
import time
import os

class BeeHeader(Packet):
    name = "BeeHeader"
    fields_desc = [
        BitField("port_idx_in_reg", 0, 31),
        BitField("queue_occ_info", 0, 1)
    ]

def send_bee_packets_dumbbell():
    switch_name = 's1'
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
    
    n_hosts = 4
    bw = 500
    delay = 0
    p4_program = 'sd/sd.p4' 
    exp_id = "0000-deflection"
    topology = DumbbellTopology(n_hosts, bw, delay, p4_program)
    #topology = LeafSpineTopology(2, 2, 2, bw, delay, p4_program)
    control_plane = SimpleDeflectionControlPlane(topology)

    topology.generate_topology()
    topology.start_network()
    control_plane.generate_control_plane()
    topology.net.program_switches()

    time.sleep(2)

    send_bee_packets_dumbbell()
    #topology.net.start_net_cli()

    client = topology.net.net.get('h1')
    server = topology.net.net.get('h2')

    time.sleep(2)

    # tcp dump
    switch = topology.net.net.get('s1')
    os.makedirs(f"tmp/{exp_id}", exist_ok=True)
    switch.cmd(f"tcpdump -i s1-eth1 > tmp/{exp_id}/eth1_deflection.log &") # to h2
    switch.cmd(f"tcpdump -i s1-eth3 > tmp/{exp_id}/eth3_deflection.log &") # to s2

    server.cmd(f"iperf3 -s > tmp/{exp_id}/iperf_server.log &")
    time.sleep(1)
    
    # run the queue logger
    os.system("python3 queue_logger.py &")

    client.cmd("iperf3 -c " + server.IP() + f" -t 5 > tmp/{exp_id}/iperf_client.log")

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
