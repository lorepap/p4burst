#!/usr/bin/env python3

"""

TODO:
    - Sender code for RL data collection (based on tx_order.py)
    - Receiver code for RL data collection (based on rx_order.py)

"""

import sys
import os
import time
import random
import traceback
import argparse
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from configparser import ConfigParser
import subprocess

from metrics import FlowMetricsManager
from topology import LeafSpineTopology, DumbbellTopology
from control_plane import ECMPControlPlane, L3ForwardingControlPlane, SimpleDeflectionControlPlane, BaseControlPlane

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.control_plane: BaseControlPlane = None
        # Path
        self.db_path = 'data/distributions'
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S") if not args.exp_id else args.exp_id
        if not self.args.disable_metrics:
            FlowMetricsManager(self.exp_id) # create a new database for tracking simulation metrics
        self.config = self.load_config()
        self.p4_program=self.set_p4_program()
        self.processes = []

    def load_config(self, config_file='config.ini'):
        config = ConfigParser()
        config.read(config_file)
        self.config = config
        # global
        self.n_hosts = self.config.getint('global', 'n_hosts')
        self.app = self.config.get('global', 'app')
        self.topology_str = self.config.get('global', 'topology')
        self.leaf = self.config.getint('leafspine', 'n_leaf')
        self.spine = self.config.getint('leafspine', 'n_spine')
        self.bw = self.config.getfloat('global', 'bw')
        self.delay = self.config.getfloat('global', 'delay')
        self.loss = self.config.getfloat('global', 'loss')
        self.control_plane_str = self.config.get('global', 'control_plane')
        self.cong_proto = self.config.get('global', 'cong_proto')
        # background
        self.n_servers = self.config.getint('background', 'n_servers')
        self.flow_multiplier = self.config.getfloat('background', 'flow_multiplier')
        self.inter_multiplier = self.config.getfloat('background', 'inter_multiplier')
        # bursty
        self.qps = self.config.getint('bursty', 'qps')
        self.incast_degree = self.config.getint('bursty', 'incast_degree')
        self.reply_size = self.config.getint('bursty', 'reply_size')
        # collect
        self.n_flows = self.config.getint('collect', 'n_flows')
        self.n_packets = self.config.getint('collect', 'n_packets')
        self.interval = self.config.getfloat('collect', 'interval')
        self.n_clients = self.config.getint('collect', 'n_clients')


    # TODO refactor
    def set_p4_program(self):
        if self.control_plane_str == 'ecmp':
            return 'ecmp.p4'
        elif self.control_plane_str == 'l3':
            return 'l3_forwarding.p4'
        elif self.control_plane_str == 'simple_deflection':
            return 'sd/sd.p4'
        else:
            raise ValueError(f"Unsupported control plane: {self.control_plane_str}")

    def setup_experiment(self):
        if self.topology_str == 'leafspine':
            self.topology = LeafSpineTopology(
                self.n_hosts, 
                self.leaf, 
                self.spine, 
                self.bw, 
                self.delay,
                self.p4_program
            )
        elif self.topology_str == 'dumbbell':
            self.topology = DumbbellTopology(self.n_hosts, self.bw, self.delay, self.p4_program)
        else:
            raise ValueError(f"Unsupported topology: {self.args.topology}")

        if self.control_plane_str == 'ecmp':
            self.control_plane = ECMPControlPlane(self.topology, self.leaf, self.spine)
        elif self.control_plane_str == 'l3':
            self.control_plane = L3ForwardingControlPlane(self.topology)
        elif self.control_plane_str == 'simple_deflection':
            self.control_plane = SimpleDeflectionControlPlane(self.topology)
        else:
            raise ValueError(f"Unsupported control plane: {self.control_plane_str}")

    def start_network(self):
        """
        Start the network and generate the control plane.
        """
        self.topology.generate_topology()
        if self.args.switch_pcap:
            self.topology.enable_switch_pcap()
        self.topology.start_network()
        self.control_plane.generate_control_plane()
        self.topology.net.program_switches() # insert the rules

    def stop_network(self):
        if self.topology.net:
            self.topology.net.stopNetwork()

    def run_bursty_app(self):
        if not self.incast_degree:
            raise ValueError("Incast degree must be specified for bursty app")
        
        server_ips = []
        client = random.choice(self.topology.net.net.hosts)
        servers = self.select_servers(n=self.incast_degree)
        
        print(f"Selected servers: {[server.name for server in servers]}")
        #os.makedirs("log/bursty_servers", exist_ok=True)

        # for server in servers:
        for server in self.topology.net.net.hosts:
            # flow_data = self.load_bursty_data(server.name, 'web_bursty', 1.0, 0.11) # TODO: use the config file for params
            if server.IP() != client.IP():
                print(f"Starting server on {server.name} ({server.IP()})...")
                server.cmd(f'python3 -m app --mode server --host_ip {server.IP()} --type bursty --reply_size {self.reply_size} --exp_id {self.exp_id} > /dev/null 2>&1 &')
                server_ips.append(server.IP())
        
        time.sleep(2)  # Give the servers some time to start up

        #os.makedirs("log/bursty_clients", exist_ok=True)
        print(f"Starting client on {client.name} ({client.IP()})...")
        #client.cmd(f'python3 -m app --mode client --type bursty --server_ips {" ".join(server_ips)} --exp_id {self.exp_id} --incast_scale {self.incast_degree} > /dev/null 2>&1 &')

        client_cmd = (
            f'python3 -m app --mode client --type bursty '
            f'--server_ips {" ".join(server_ips)} '
            f'--exp_id {self.exp_id} '
            f'--incast_scale {self.incast_degree} '
            f'--duration {self.args.duration}'
        )

        print(f"Starting client: {client_cmd}")
        proc = client.popen(client_cmd, shell=True, stderr=sys.stderr, stdout=subprocess.DEVNULL)
        self.processes.append(proc)


    def run_background_app(self):
        self.bg_servers = self.topology.net.net.hosts[:self.n_servers] # TODO: we should randomize the servers -> requires db creation for arbitrary number of servers (<320)
        print(f"Servers: {[server.name for server in self.bg_servers]}")

        # Start the server processes on each host.
        for server in self.bg_servers:
            print(f"Starting server on {server.name} ({server.IP()})...")
            # Launch in background since we might not need to wait on these.
            server.cmd(f'python3 -m app --mode server --type background --host_ip {server.IP()} --exp_id {self.exp_id} > /dev/null 2>&1 &')

        time.sleep(1)  # Allow servers to start

        # For each server (or whichever host is acting as a client), start the background client.
        for server in self.bg_servers:
            flow_data = self.load_background_flow_data(server.name, 'cache', self.flow_multiplier, self.inter_multiplier)
            # Extract parameters from flow_data
            server_ids = [flow['server_idx'] for flow in flow_data]
            server_names = [f'h{int(server_id)+1}' for server_id in server_ids]
            server_ips = [self.topology.net.net.get(server_name).IP() for server_name in server_names]
            flow_ids = [flow['flow_id'] for flow in flow_data]
            flow_sizes = [flow['flow_size'] for flow in flow_data]
            inter_arrival_times = [flow['inter_arrival_time'] for flow in flow_data]

            # Construct the client command, now including a duration parameter.
            client_cmd = (
                f'python3 -m app --mode client '
                f'--server_ips {" ".join(server_ips)} '
                f'--flow_ids {" ".join(map(str, flow_ids))} '
                f'--flow_sizes {" ".join(map(str, flow_sizes))} '
                f'--iat {" ".join(map(str, inter_arrival_times))} '
                f'--type background '
                f'--duration {self.args.duration} '
                f'--exp_id {self.exp_id} '
                f'--congestion_control {self.cong_proto}'
            )
            print(f"Starting background client on {server.name}: {client_cmd}")

            # Use popen() so we can wait for the client process to finish.
            proc = server.popen(client_cmd, shell=True, stderr=sys.stderr, stdout=subprocess.DEVNULL)
            self.processes.append(proc)

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
    
        # Start server on h3
        print(f"Starting server on h3 ({h3.IP()})...")
        h3.cmd(f'python3 -m app --mode server --host_ip {h3.IP()} --type single &')
        time.sleep(2)
        print(f"Starting client on h1 ({h1.IP()})...")
        h1.cmd(f'python3 -m app --mode client --type single --server_ips {h3.IP()} &')

    def run_iperf_app(self):
        # Two clients send iperf flows to two servers
        h1 = self.topology.net.net.get('h1')
        h2 = self.topology.net.net.get('h2')
        # h3 = self.topology.net.net.get('h3')
        # h4 = self.topology.net.net.get('h4')

        # for logging purposes
        os.makedirs(f'/home/ubuntu/p4burst/tmp/{self.exp_id}', exist_ok=True)

        # Capture senders pcap
        pcap_dir = 'pcap'
        for host in self.topology.net.net.hosts:
            for intf in host.intfList():
                if intf.name != 'lo':
                    cmd = f'tcpdump -i {intf} -w {pcap_dir}/{intf}.pcap &'
                    host.cmd(cmd)

        # Start servers
        print(f"Starting server on h2 ({h2.IP()})...")
        h2.cmd(f'iperf3 -s --logfile /home/ubuntu/p4burst/tmp/{self.exp_id}/iperf_app_server_h3.log &')
        #print(f"Starting server on h4 ({h4.IP()})...")
        #h4.cmd(f'iperf3 -s -p 5202 --logfile /home/ubuntu/p4burst/tmp/{self.exp_id}/iperf_app_server_h4.log &')
        time.sleep(2)

        # Start clients
        print(f"Running client on h1 ({h1.IP()}) for {self.args.duration}s...")
        cmd = f'iperf3 -c {h2.IP()} -t {self.args.duration} -p 5201 --logfile /home/ubuntu/p4burst/tmp/{self.exp_id}/iperf_app_client_h1.log'
        proc = h1.popen(cmd, shell=True, stderr=sys.stderr, stdout=subprocess.DEVNULL)
        self.processes.append(proc)

        # print(f"Starting client on h2 ({h2.IP()}) for {self.args.duration}s...")
        # h2.cmd(f'iperf3 -c {h4.IP()} -t {self.args.duration} -p 5202 --logfile /home/ubuntu/p4burst/tmp/{self.exp_id}/iperf_app_client_h2.log &')
  
    def select_servers(self, n):
        # Check if the number of servers requested is greater than the available servers
        if n >= len(self.topology.net.net.hosts):
            raise ValueError("Number of servers requested must be less than the total number of hosts.")
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
                        f"{len(self.bg_servers)}_servers.db"

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
    
    def run_mixed_app(self):
        try:
            self.run_background_app()
            self.run_bursty_app()
        except Exception as e:
            traceback.print_exc()
            raise e

    def run_collection(self):
        from utils.rl_data_utils import collect_switch_logs, combine_datasets
        exp_dir = f'tmp/{self.exp_id}'
        receiver_logs = []
        
        # Client - Server logic
        clients = []
        clients.append(self.topology.net.net.get('h1'))

        client_cmd = (
                f'python3 -m app --mode client '
                f'--type collect '
                f'--server_ip {self.topology.net.net.get("h2").IP()} '
                f'--num_packets {self.n_packets} '
                f'--interval {self.interval}'
                f'--num_flows {self.n_flows}'
                f'--congestion_control {self.cong_proto} '
                f'--exp_id {self.exp_id}' # to integrate in the collection client
            )

        # start server
        print(f"Starting server on {self.topology.net.net.get('h2').name}...")
        server = self.topology.net.net.get('h2')
        server_csv_file = f"{exp_dir}/receiver_log.csv"
        server.cmd(f'python3 -m app --mode server --type collect --ip {server.IP()} --exp_id {self.exp_id} --log_file {server_csv_file}> /dev/null 2>&1 &')

        for client in clients:
            print(f"Starting client on {client.name}...")
            proc = client.popen(client_cmd, shell=True, stderr=sys.stderr, stdout=subprocess.DEVNULL)
            self.processes.append(proc)
            
        # Create the dataset
        for i, client in enumerate(clients):
            server_receiver_log = f"{exp_dir}/receiver_log_{i+1}.csv"
            receiver_logs.append(server_receiver_log)
        
        switch_datasets = collect_switch_logs(self.topology, exp_dir)
        final_dataset = combine_datasets(switch_datasets, receiver_logs, exp_dir)

    def run_experiment(self):
        exp_dict = {
                'mixed': self.run_mixed_app,
                'bursty': self.run_bursty_app,
                'background': self.run_background_app,
                'simple': self.run_simple_app,
                'iperf': self.run_iperf_app,
                'collect': self.run_collection
            }
        
        try:
            self.setup_experiment()
            self.start_network() # will run cli if specified

            # Pre-run setup
            if isinstance(self.control_plane, SimpleDeflectionControlPlane):
                # Note: I add the bee packets logic to the SD control plane to keep the runner clean
                for switch in self.topology.get_leaf_switches():
                    print("Sending BEE packets to switch", switch)
                    self.control_plane.send_bee_packets(switch=switch)
                # Run the queue logger (debug)
                os.system(f"python3 queue_logger.py --log tmp/{self.exp_id}/queue_log.txt &")
            
            if self.args.cli:
                self.topology.net.start_net_cli()  # debugging

            if self.args.host_pcap:
                self.enable_pcap_hosts()
            
            if self.app:
                exp_dict[self.app]()
            
            else:
                # run the CLI
                print("No app specified. Running Mininet CLI...")
                self.topology.net.start_net_cli()
            
            print("Waiting for processes to finish...")
            #time.sleep(self.args.duration + 5)
            
            for proc in self.processes:
                proc.wait(timeout=self.args.duration + 5)

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
    parser.add_argument('--duration', '-d', type=int, default=10, help='Duration of the experiment in seconds')
    parser.add_argument('--host_pcap', action='store_true', help='Enable pcap on hosts')
    parser.add_argument('--switch_pcap', action='store_true', help='Enable pcap on switches')
    parser.add_argument('--cli', action='store_true', help='Enable Mininet CLI')
    parser.add_argument('--disable_metrics', action='store_true', help='Disable metrics collection')
    parser.add_argument('--exp_id', type=str, default=None, help='Experiment ID')
    return parser.parse_args()

def main():
    args = get_args()
    experiment = ExperimentRunner(args)
    experiment.run_experiment()

if __name__ == "__main__":
    main()