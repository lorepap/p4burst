#!/usr/bin/env python3

import argparse
import os
import time
import random
import traceback
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

from metrics import FlowMetricsManager
from topology import LeafSpineTopology, DumbbellTopology
from control_plane import ECMPControlPlane, L3ForwardingControlPlane, SimpleDeflectionControlPlane, TestControlPlane

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.topology = None
        self.control_plane = None
        # Path
        self.db_path = 'data/distributions'
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        flowtracker = FlowMetricsManager(self.exp_id) # create a new database for tracking simulation metrics
        self.p4_program=self.set_p4_program(args.control_plane)

    # TODO refactor
    def set_p4_program(self, control_plane):
        if control_plane == 'ecmp':
            return 'ecmp.p4'
        elif control_plane == 'l3':
            return 'l3_forwarding.p4'
        elif control_plane == 'simple_deflection':
            return 'Simple_Deflection/sd.p4'
        else:
            raise ValueError(f"Unsupported control plane: {control_plane}")

    def setup_experiment(self):
        if self.args.topology == 'leafspine':
            self.topology = LeafSpineTopology(
                self.args.hosts, 
                self.args.leaf, 
                self.args.spine, 
                self.args.bw, 
                self.args.latency,
                self.p4_program
            )
        elif self.args.topology == 'dumbbell':
            self.topology = DumbbellTopology(self.args.hosts, self.args.bw, self.args.latency, self.p4_program)
        else:
            raise ValueError(f"Unsupported topology: {self.args.topology}")

        if self.args.control_plane == 'ecmp':
            self.control_plane = ECMPControlPlane(self.topology, self.args.leaf, self.args.spine)
        elif self.args.control_plane == 'l3':
            self.control_plane = L3ForwardingControlPlane(self.topology)
        elif self.args.control_plane == 'simple_deflection':
            # self.control_plane = SimpleDeflectionControlPlane(self.topology)
            self.control_plane = SimpleDeflectionControlPlane(self.topology)
        else:
            raise ValueError(f"Unsupported control plane: {self.args.control_plane}")

    def start_network(self):
        self.topology.generate_topology()
        if self.args.switch_pcap:
            self.topology.enable_switch_pcap()
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
        client = self.topology.net.net.get('h1') # TODO generalize with more clients
        servers = self.select_servers(n=self.args.incast_degree)
        
        print(f"Selected servers: {[server.name for server in servers]}")
        os.makedirs("log/bursty_servers", exist_ok=True)

        # for server in servers:
        for server in self.topology.net.net.hosts:
            # flow_data = self.load_bursty_data(server.name, 'web_bursty', 1.0, 0.11) # TODO: use the config file for params
            if server.IP() != client.IP():
                print(f"Starting server on {server.name} ({server.IP()})...")
                server.cmd(f'python3 -m app --mode server --host_ip {server.IP()} --type bursty --reply_size {self.args.reply_size} --exp_id {self.exp_id} > log/bursty_servers/{server.name}.log 2>&1 &')
                server_ips.append(server.IP())
        
        time.sleep(2)  # Give the servers some time to start up

        os.makedirs("log/bursty_clients", exist_ok=True)
        print(f"Starting client on {client.name} ({client.IP()})...")
        client.cmd(f'python3 -m app --mode client --type bursty --server_ips {" ".join(server_ips)} --exp_id {self.exp_id} --incast_scale {self.args.incast_degree} > log/bursty_clients/{client.name}.log 2>&1 &')

    def run_background_app(self):
        servers = self.topology.net.net.hosts
        print(f"Servers: {[server.name for server in servers]}")
        
        os.makedirs("log/background_servers", exist_ok=True)
        # Starting servers
        for server in servers:
            print(f"Starting server on {server.name} ({server.IP()})...")
            server.cmd(f'python3 -m app --mode server --type background --host_ip {server.IP()} --exp_id {self.exp_id} > log/background_servers/{server.name}.log 2>&1 &')
        
        time.sleep(1)
        
        os.makedirs("log/background_clients", exist_ok=True)
        # Starting clients
        for server in  servers:
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
            print(f"[DEBUG] Client {host_id} flow ids: {flow_ids}")
            server.cmd(f'python3 -m app --mode client \
                --server_ips {" ".join(server_ips)} \
                --flow_ids {" ".join(map(str, flow_ids))} \
                --flow_sizes {" ".join(map(str, flow_sizes))} \
                --iat {" ".join(map(str, inter_arrival_times))} \
                --type background \
                --exp_id {self.exp_id} > log/background_clients/{server.name}.log 2>&1 &') # I'm not convinced how server_ips are passed to the client

    def run_simple_app(self):
        """
        Run a simple client-server application that sends one single packet from a client to a server.
        """
        # Get hosts
        h1 = self.topology.net.net.get('h1')
        h3 = self.topology.net.net.get('h3')

        # Set congestion control
        h1.cmd('sysctl -w net.ipv4.tcp_congestion_control=cubic')
    
        # h1.cmd('ifconfig h1-eth0 txqueuelen 1000')
        os.makedirs("log/simple_servers", exist_ok=True)
        # Start server on h3
        print(f"Starting server on h3 ({h3.IP()})...")
        h3.cmd(f'python3 -m app --mode server --host_ip {h3.IP()} --type single > log/simple_servers/h3.log 2>&1 &')
        time.sleep(2)
        os.makedirs("log/simple_clients", exist_ok=True)
        print(f"Starting client on h1 ({h1.IP()})...")
        h1.cmd(f'python3 -m app --mode client --type single --server_ips {h3.IP()} > log/simple_clients/h1.log 2>&1 &')

    def run_iperf_app(self):
        # Two clients send iperf flows to two servers
        h1 = self.topology.net.net.get('h1')
        h2 = self.topology.net.net.get('h2')
        h3 = self.topology.net.net.get('h3')
        h4 = self.topology.net.net.get('h4')

        # Capture senders pcap
        pcap_dir = 'pcap'
        for host in [h1, h2, h3]:
            for intf in host.intfList():
                if intf.name != 'lo':
                    cmd = f'tcpdump -i {intf} -w {pcap_dir}/{intf}.pcap &'
                    host.cmd(cmd)

        os.makedirs("log/iperf_servers", exist_ok=True)
        # Start servers
        print(f"Starting server on h3 ({h3.IP()})...")
        h3.cmd('iperf3 -s > log/iperf_servers/h3.log 2>&1 &')
        print(f"Starting server on h4 ({h4.IP()})...")
        h4.cmd('iperf3 -s -p 5202 > log/iperf_servers/h4.log 2>&1 &')
        time.sleep(2)

        os.makedirs("log/iperf_clients", exist_ok=True)
        # Start clients
        print(f"Starting client on h1 ({h1.IP()})...")
        h1.cmd(f'iperf3 -c {h3.IP()} -t {self.args.duration} -p 5201 > log/iperf_clients/h1.log 2>&1 &')
        print(f"Starting client on h2 ({h2.IP()})...")
        h2.cmd(f'iperf3 -c {h3.IP()} -t {self.args.duration} -p 5202 > log/iperf_clients/h2.log 2>&1 &')
  
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
        for (flow_id, inter_arrival_time, flow_size, server_idx) in zip(data['flow_ids'], 
                        data["inter_arrival_times"], data["flow_sizes"], data["server_indices"]):
            flow_data.append({
                'flow_id': flow_id,
                'inter_arrival_time': inter_arrival_time,
                'flow_size': flow_size,
                'server_idx': server_idx  # Destination server
            })

        return flow_data

    def run_experiment(self):
        
        exp_dict = {
                'bursty': self.run_bursty_app,
                'background': self.run_background_app,
                'simple': self.run_simple_app,
                'iperf': self.run_iperf_app
            }
        
        try:
            self.setup_experiment()
            self.start_network() # will run cli if specified
            if self.args.host_pcap:
                self.enable_pcap_hosts()
            if self.args.app:
                exp_dict[self.args.app]()
            else: # Run experiment with background + incast
                self.run_background_app()
                self.run_bursty_app()

            # Let the experiment run for a specified duration
            print(f"Experiment running for {self.args.duration+5} seconds...")
            time.sleep(self.args.duration)

        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            self.stop_network()

    def enable_pcap_hosts(self):
        pcap_dir = 'pcap'
        if not os.path.exists(pcap_dir):
            os.makedirs(pcap_dir)

        for host in self.topology.net.net.hosts:
            for intf in host.intfList():
                if intf.name != 'lo':
                    cmd = f'tcpdump -i {intf} -w {pcap_dir}/{intf}.pcap &'
                    host.cmd(cmd)


def get_args():
    parser = argparse.ArgumentParser(description='Run network experiment')
    parser.add_argument('--topology', '-t', type=str, required=True, choices=['leafspine', 'dumbbell'], help='Topology type')
    parser.add_argument('--control_plane', '-c', type=str, required=False, choices=['ecmp', 'l3', 'simple_deflection'], help='Control plane protocol', default='ecmp')
    parser.add_argument('--hosts', '-n', type=int, required=True, help='Number of hosts')
    parser.add_argument('--leaf', '-l', type=int, help='Number of leaf switches (for leaf-spine topology)', default=2)
    parser.add_argument('--spine', '-s', type=int, help='Number of spine switches (for leaf-spine topology)', default=2)
    parser.add_argument('--bw', '-b', type=int, help='Bandwidth in Mbps', default=1000)
    parser.add_argument('--latency', '-d', type=float, help='Latency in ms', default=0.1)
    parser.add_argument('--reply_size', type=int, default=40000, help='Size of the burst response in bytes')
    parser.add_argument('--incast_degree', type=int, default=5, help='Number of bursty servers')
    parser.add_argument('--duration', type=int, default=10, help='Duration of the experiment in seconds')
    parser.add_argument('--app', type=str, required=False, choices=['bursty', 'background', 'simple', 'iperf'], help='Type of application')
    parser.add_argument('--host_pcap', action='store_true', help='Enable pcap on hosts')
    parser.add_argument('--switch_pcap', action='store_true', help='Enable pcap on switches')
    parser.add_argument('--cli', action='store_true', help='Enable Mininet CLI')
    return parser.parse_args()

def main():
    args = get_args()
    experiment = ExperimentRunner(args)
    experiment.run_experiment()

if __name__ == "__main__":
    main()