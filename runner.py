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
import sqlite3
import pandas as pd
import numpy as np
from metrics import FlowMetricsManager

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.topology = None
        self.control_plane = None
        # Path
        self.db_path = 'data/distributions'
        self.flowtracker = FlowMetricsManager()

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

    def run_background_app(self):
        servers = self.topology.net.net.hosts
        print(f"Servers: {[server.name for server in servers]}")
        for server in servers:
            print(f"Starting server on {server.name} ({server.IP()})...")
            flow_data = self.load_background_flow_data(server.name, 'cache', 1.0, 11.85) # TODO: use the config file for params
            # params for client traffic -> this looks bad TODO improvements
            # this block should go somewhere else and the params shoul be passed differently
            # I also have doubts about this app runner function - TODO investigate integrating client and server directly here
            server_ids = [flow['server_idx'] for flow in flow_data]
            server_names = [f'h{int(server_id)+1}' for server_id in server_ids]
            server_ips = [self.topology.net.net.get(server_name).IP() for server_name in server_names]
            flow_ids = [flow['flow_id'] for flow in flow_data]
            flow_sizes = [flow['flow_size'] for flow in flow_data]
            inter_arrival_times = [flow['inter_arrival_time'] for flow in flow_data]
            host_id = server.name # parsed in the client
            print(f"[DEBUG] Client {host_id} Server IPs: {server_ips} | Server IDs: {server_ids}")
            server.cmd(f'python3 -m app --mode server --type background &')
            server.cmd(f'python3 -m app --mode client \
                --server_ips {" ".join(server_ips)} \
                --flow_ids {" ".join(map(str, flow_ids))} \
                --flow_sizes {" ".join(map(str, flow_sizes))} \
                --iat {" ".join(map(str, inter_arrival_times))} \
                --type background &') # I'm not convinced how server_ips are passed to the client

  
    def select_servers(self, n):
        return random.sample(self.topology.net.net.hosts, n)

    def load_background_flow_data(self, host_id, bg_application_category, 
                                bg_flow_multiplier, bg_inter_multiplier):
        """
        """
        flow_data = []

        # Determine the column name based on the host ID
        server_column = f"server{int(host_id[1:])-1}app1"  # Convert host_id hX to server(X-1)app0

        # Construct the file paths based on the specified format
        db_params = {
            "server_indices": "server_idx",
            "inter_arrival_times": "inter_arrival_time",
            "flow_sizes": "flow_size",
            "flow_ids": "flow_ids"
        }

        db_tables = {
            "server_indices": "server_idx",
            "inter_arrival_times": "inter_arrival",
            "flow_sizes": "flow_size",
            "flow_ids": "flow_ids"
        }

        # Load each parameter from its corresponding database file
        data = {}
        for (param_name, db_name), db_col in zip(db_params.items(), db_tables.values()):
            file_path = f"{bg_application_category}_background_{db_name}_db_" \
                        f"{float(bg_flow_multiplier):.6f}_flowmult_" \
                        f"{float(bg_inter_multiplier):.6f}_intermult_0_" \
                        f"{self.args.hosts}_servers.db"

            if not os.path.exists(os.path.join(self.db_path, file_path)):
                raise ValueError(f"File {file_path} not found in {self.db_path}")
            
            try:
                conn = sqlite3.connect(os.path.join(self.db_path, file_path))
                print(file_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_name = cursor.fetchone()[0]

                # Retrieve only the specific column for this server's host id
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                conn.close()
            except:
                raise ValueError(f"Error loading data from {file_path}")

            # Retrieve the specific column for this server's host ID
            if server_column in df.columns:
                data[param_name] = df[server_column].replace('', np.nan).dropna().tolist()
            else:
                raise ValueError(f"Column {server_column} not found in {file_path}")
            
        # Combine the data from each database into flow data entries
        for flow_id, (inter_arrival_time, flow_size, server_idx) in enumerate(
                zip(data["inter_arrival_times"], data["flow_sizes"], data["server_indices"])):
            flow_data.append({
                'flow_id': flow_id,
                'inter_arrival_time': inter_arrival_time,
                'flow_size': flow_size,
                'server_idx': server_idx  # Destination server
            })

        return flow_data

    def run_experiment(self):
        try:
            self.setup_experiment()
            self.start_network() # will run cli if specified
            # self.run_bursty_app()
            self.run_background_app()            
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
    parser.add_argument('--latency', '-d', type=float, help='Latency in ms', default=10)
    parser.add_argument('--reply_size', type=int, default=40000, help='Size of the burst response in bytes')
    parser.add_argument('--n_bursty_servers', type=int, default=2, help='Number of bursty servers')
    parser.add_argument('--duration', type=int, default=10, help='Duration of the experiment in seconds')
    parser.add_argument('--cli', action='store_true', help='Enable Mininet CLI')
    return parser.parse_args()

def main():
    args = get_args()
    experiment = ExperimentRunner(args)
    experiment.run_experiment()

if __name__ == "__main__":
    main()