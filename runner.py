#!/usr/bin/env python3

import argparse
import os
import time
import random
from p4utils.mininetlib.network_API import NetworkAPI
from topology import LeafSpineTopology, DumbbellTopology
from control_plane import LeafSpineControlPlane, DumbbellControlPlane
from app import BurstyServer, BurstyClient
import traceback

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.topology = None
        self.control_plane = None

    def setup_experiment(self):
        if self.args.topology == 'leafspine':
            self.topology = LeafSpineTopology(
                self.args.hosts, 
                self.args.leaf, 
                self.args.spine, 
                self.args.bw, 
                self.args.latency
            )
            self.control_plane = LeafSpineControlPlane(self.topology.net, self.args.leaf, self.args.spine)
        elif self.args.topology == 'dumbbell':
            self.topology = DumbbellTopology(self.args.hosts, self.args.bw, self.args.latency)
            self.control_plane = DumbbellControlPlane(self.topology.net)
        else:
            raise ValueError(f"Unsupported topology: {self.args.topology}")

    def start_network(self):
        self.topology.generate_topology()
        self.topology.start_network()
        self.control_plane.generate_control_plane()
        self.topology.net.program_switches() # insert the rules
        if self.args.cli:
            self.topology.net.start_net_cli()  # debugging

    def stop_network(self):
        if self.topology.net:
            self.topology.net.stopNetwork()

    def run_bursty_app(self):
        server_ips = []
        client = self.topology.net.net.get('h1')
        servers = self.select_servers(n=self.args.n_bursty_servers)
        
        print(f"Selected servers: {[server.name for server in servers]}")
        
        for server in servers:
            if server.IP() != client.IP():
                print(f"Starting server on {server.name} ({server.IP()})...")
                server.cmd(f'python3 -m app --mode server --type bursty --reply_size {self.args.reply_size} &')
                server_ips.append(server.IP())
        
        time.sleep(2)  # Give the servers some time to start up

        print(f"Starting client on {client.name} ({client.IP()})...")
        client.cmd(f'python3 -m app --mode client --type bursty --server_ips {" ".join(server_ips)} &')

    def select_servers(self, n):
        return random.sample(self.topology.net.net.hosts, n)

    def run_experiment(self):
        try:
            self.setup_experiment()
            self.start_network()
            self.run_bursty_app()
            
            # Let the experiment run for a specified duration
            print(f"Experiment running for {self.args.duration} seconds...")
            time.sleep(self.args.duration)

        except Exception as e:
            # traceback.print_exc()
            raise e
        finally:
            self.stop_network()

def get_args():
    parser = argparse.ArgumentParser(description='Run network experiment')
    parser.add_argument('--topology', '-t', type=str, required=True, choices=['leafspine', 'dumbbell'], help='Topology type')
    parser.add_argument('--hosts', '-n', type=int, required=True, help='Number of hosts')
    parser.add_argument('--leaf', '-l', type=int, help='Number of leaf switches (for leaf-spine topology)', default=2)
    parser.add_argument('--spine', '-s', type=int, help='Number of spine switches (for leaf-spine topology)', default=2)
    parser.add_argument('--bw', '-b', type=int, help='Bandwidth in Mbps', default=10)
    parser.add_argument('--latency', '-d', type=int, help='Latency in ms', default=10)
    parser.add_argument('--reply_size', type=int, default=40000, help='Size of the burst response in bytes')
    parser.add_argument('--n_bursty_servers', type=int, default=2, help='Number of bursty servers')
    parser.add_argument('--duration', type=int, default=60, help='Duration of the experiment in seconds')
    parser.add_argument('--cli', action='store_true', help='Enable Mininet CLI')
    return parser.parse_args()

def main():
    args = get_args()
    experiment = ExperimentRunner(args)
    experiment.run_experiment()

if __name__ == "__main__":
    main()