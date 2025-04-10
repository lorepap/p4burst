#!/usr/bin/env python3

"""
Collection Runner - Simplified runner for packet collection experiments

This script runs packet collection experiments using UDP flows with custom headers
for reordering detection and analysis, specifically using the SimpleDeflection control plane.


TODO: 

- aggregate metrics for the reward (FCT, delay)
- TCP support (replace UDP with TCP)

"""

import sys
import logging
from datetime import datetime
import os
import utils.rl_data_utils as datalib

# Set up root logger first thing
def setup_logging(exp_id):
    # Clear any existing handlers from the root logger
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join('tmp', exp_id, "collection_runner.log"))
        ]
    )
    
    # Return a logger for this module
    return logging.getLogger("CollectionRunner")

exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = f'tmp/{exp_id}'
os.makedirs(exp_dir, exist_ok=True)
logger = setup_logging(exp_id)

# Rest of your imports
import time
import random
import traceback
import argparse
import subprocess

from topology import LeafSpineTopology, DumbbellTopology
from control_plane import RLDeflectionControlPlane, SimpleDeflectionControlPlane
# from metrics import FlowMetricsManager
# from utils.rl_data_utils import collect_switch_logs, combine_datasets

# Add a sanity check log to verify logging is working
logger.info("Collection Runner starting - logging is active")

p4_program_paths = {
    'simple_deflection': 'Simple_Deflection/sd.p4',
    'ecmp': 'ecmp.p4',
    'dist_preemptive_deflection': 'Dist_PD/distpd.p4',
    'quantile_preemptive_deflection': 'Quantile_PD/quantilepd.p4',
    'rl_deflection': 'evaluation/evaluation.p4'
}

p4_const_paths = {
    'simple_deflection': 'SimpleDeflection/includes/sd_const.p4',
    'ecmp': 'ecmp_const.p4',
    'dist_preemptive_deflection': 'Dist_PD/includes/distpd_const.p4',
    'quantile_preemptive_deflection': 'Quantile_PD/includes/quantilepd_const.p4',
    'rl_deflection': 'evaluation/includes/evaluation_const.p4'
}

p4_control_plane = {
    'simple_deflection': SimpleDeflectionControlPlane,
    'rl_deflection': RLDeflectionControlPlane,
}


class CollectionRunner:
    """
    Runner specifically for collection experiments that analyze packet reordering.
    Uses SimpleDeflection control plane.
    """
    def __init__(self, args):
        self.args = args
        self.exp_id = exp_id
        self.exp_dir = f'tmp/{exp_id}'
        
        # Configure experiment parameters - fixed to simple deflection
        self.processes = []
        self.topology_type = 'leafspine'  # Only using leaf-spine topology
        self.n_hosts = args.n_hosts
        self.n_leaf = args.n_leaf
        self.n_spine = args.n_spine
        self.bw = args.bw
        self.delay = args.delay
        self.policy = args.policy
        self.queue_rate = args.queue_rate
        self.queue_depth = args.queue_depth
        
        # Collection parameters
        self.n_clients = args.n_clients
        self.n_servers = args.n_servers
        #self.num_flows = args.num_flows
        self.flow_iat = args.flow_iat
        self.flow_size = args.flow_size
        self.congestion_control = args.congestion_control
        self.burst_reply_size = args.bursty_reply_size
        self.burst_interval = args.burst_interval
        self.burst_servers = args.burst_servers
        self.burst_clients = args.burst_clients
        
        # Initialize flow metrics if tracking enabled
        # if not args.disable_metrics:
        #     self.flow_metrics = FlowMetricsManager(self.exp_id)
            
        logger.info(f"Initialized collection runner with experiment ID: {self.exp_id}")

    def setup_experiment(self):
        """Set up the experiment topology with SimpleDeflection control plane."""
        logger.info(f"Setting up leaf-spine topology with SimpleDeflection control plane")
        
        # Create topology - fixed to leaf-spine for simple deflection
        self.topology = LeafSpineTopology(
            self.n_hosts, 
            self.n_leaf, 
            self.n_spine, 
            self.bw, 
            self.delay,
            p4_program_paths[self.policy]
        )
        
        # Create control plane - fixed to SimpleDeflection
        self.control_plane = p4_control_plane[self.policy](self.topology, queue_rate=self.queue_rate, 
                                                          queue_depth=self.queue_depth)
        
        logger.info("Experiment setup complete")

    def start_network(self):
        """Start the network and generate the control plane."""
        logger.info("Starting network...")
        self.topology.generate_topology()
        
        # Enable packet captures if requested
        if self.args.switch_pcap:
            self.topology.enable_switch_pcap()
        
        # OLD: moved to the server/client classes
        # if self.args.host_pcap:
        #     self.enable_pcap_hosts()
            
        self.topology.start_network()
        self.control_plane.generate_control_plane()
        self.topology.net.program_switches()  # insert the rules
        logger.info("Network started")

    def stop_network(self):
        """Stop the network."""
        if hasattr(self, 'topology') and self.topology.net:
            logger.info("Stopping network...")
            self.topology.net.stopNetwork()
            logger.info("Network stopped")

    # def enable_pcap_hosts(self):
    #     """Enable packet capture on host interfaces."""
    #     pcap_dir = f'{self.exp_dir}/pcap'
    #     os.makedirs(pcap_dir, exist_ok=True)

    #     for host in self.topology.net.net.hosts:
    #         for intf in host.intfList():
    #             if (intf.name != 'lo') and (intf.name != 'eth0'):
    #                 cmd = f'tcpdump -i {intf} -w {pcap_dir}/{intf}.pcap &'
    #                 host.cmd(cmd)
    #     logger.info("Host packet capture enabled")

    def run_test_collection(self):
        """Run the collection experiment."""
        logger.info("Starting collection experiment...")
        receiver_logs = []
        
        # Select client and server hosts

        # Select hosts attached to switch s1
        s1 = self.topology.net.net.get('s1')
        s1_hosts = [h for h in self.topology.net.net.hosts if self.topology.net.areNeighbors(s1.name, h.name)]
        if self.n_clients > len(s1_hosts):
            raise ValueError("Number of clients exceeds number of hosts on s1")

        client_hosts = []
        for i in range(self.n_clients):
            client_hosts.append(s1_hosts[i])
            
        server_host = self.topology.net.net.get('h2')
        
        # Prepare log file path
        server_csv_file = f"{self.exp_dir}/receiver_log.csv"
        
        # Start the server
        logger.info(f"Starting collection server on {server_host.name} ({server_host.IP()})...")
        server_cmd = (
            'python3 -m app --mode server '
            f'--exp_id {self.exp_id} '
            '--type collect '
            '--port 12345 '
            f'--server_ips {server_host.IP()} '
            f'--server_csv_file {server_csv_file} > {self.exp_dir}/server_out.log 2>&1 &'
        )
        server_host.cmd(server_cmd)
        # Give server time to initialize
        time.sleep(1)
        
        # Start the clients
        for client_host in client_hosts:
            logger.info(f"Starting collection client on {client_host.name} ({client_host.IP()})...")
            client_cmd = (
                'python3 -m app '
                '--mode client '
                f'--exp_id {self.exp_id} '
                '--type collect '
                f'--server_ips {server_host.IP()} '
                f'--bg_flow_iat {self.flow_iat} '
                #f'--num_flows {self.num_flows} '
                f'--congestion_control {self.congestion_control} '
                f'--flow_size {self.flow_size} '
                
            )
            proc = client_host.popen(client_cmd, shell=True, stderr=sys.stderr, stdout=subprocess.DEVNULL)
            self.processes.append(proc)
        
        # Wait for client to finish
        logger.info(f"Waiting for collection to complete (duration: {self.args.duration}s)...")
        time.sleep(self.args.duration + 2)
        
        # Add the receiver log to the list
        receiver_logs.append(server_csv_file)
        logger.info("Collection experiment completed")
        
        # Return the receiver logs for dataset generation later
        return receiver_logs

    def run_collection(self):
        """Run the traffic collection experiment with separate background and bursty traffic."""
        logger.info("Starting TCP traffic experiment...")
        receiver_logs = []
        
        # Use available hosts as servers
        hosts = self.topology.net.net.hosts
        # Random servers
        servers = random.sample(hosts, self.n_servers)
        # Random clients
        clients = random.sample([h for h in hosts if h not in servers], self.n_clients)
        
        # Start all servers - both background and burst
        for i, server_host in enumerate(servers):
            logger.info(f"Starting background TCP server on {server_host.name} ({server_host.IP()})...")
            bg_server_cmd = (
                'python3 -m app --mode server '
                f'--exp_id {self.exp_id} '
                '--type collect '
                '--traffic_type background '
                '--port 12345 '
                f'--server_ips {server_host.IP()} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""} '
                f'> {self.exp_dir}/bg_server_{server_host.name}_out.log 2>&1 &'
            )
            server_host.cmd(bg_server_cmd)
            
            # Burst TCP server
            logger.info(f"Starting burst TCP server on {server_host.name} ({server_host.IP()})...")
            burst_server_cmd = (
                'python3 -m app --mode server '
                f'--exp_id {self.exp_id} '
                '--type collect '
                '--traffic_type burst '
                '--port 12346 '
                f'--server_ips {server_host.IP()} '
                f'--burst_reply_size {self.burst_reply_size} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""} '
                f'> {self.exp_dir}/burst_server_{server_host.name}_out.log 2>&1 &'
            )
            server_host.cmd(burst_server_cmd)
        
        # Give servers time to initialize
        time.sleep(2)

        # Start clients - ALL clients run background traffic, but only a subset runs bursty traffic
        client_csv_files = []
        
        # Determine how many clients will be bursty (random subset)
        # Add a new parameter to control this or use a fixed percentage
        num_bursty_clients = min(self.burst_clients, self.n_clients) if hasattr(self.args, 'bursty_clients') else max(1, self.n_clients // 2)
        bursty_clients = random.sample(clients, num_bursty_clients)
        logger.info(f"Selected {num_bursty_clients}/{self.n_clients} clients to generate bursty traffic")
        
        for i, client_host in enumerate(clients):
            # Background TCP client - ALL clients run this
            bg_client_file = f"{self.exp_dir}/bg_client_{client_host.name}_log.csv"
            client_csv_files.append(bg_client_file)
            
            logger.info(f"Starting background TCP client on {client_host.name} ({client_host.IP()})...")
            server_ips = ' '.join([server.IP() for server in servers])
            
            bg_client_cmd = (
                'python3 -m app '
                '--mode client '
                f'--exp_id {self.exp_id} '
                '--type collect '
                '--traffic_type background '
                f'--server_ips {server_ips} '
                f'--bg_flow_iat {self.flow_iat} '
                f'--flow_size {self.flow_size} '
                f'--duration {self.args.duration} '
                f'--client_csv_file {bg_client_file} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""} '
                f'> {self.exp_dir}/bg_client_{client_host.name}_out.log 2>&1 &'
            )
            proc = client_host.popen(bg_client_cmd, shell=True)
            self.processes.append(proc)
            
            # Burst TCP client - ONLY a subset of clients run this
            if client_host in bursty_clients:
                burst_client_file = f"{self.exp_dir}/burst_client_{client_host.name}_log.csv"
                client_csv_files.append(burst_client_file)
                
                logger.info(f"Starting burst TCP client on {client_host.name} ({client_host.IP()})...")
                
                burst_client_cmd = (
                    'python3 -m app '
                    '--mode client '
                    f'--exp_id {self.exp_id} '
                    '--type collect '
                    '--traffic_type burst '
                    f'--server_ips {server_ips} '
                    f'--burst_interval {self.burst_interval} '
                    f'--burst_servers {self.burst_servers} '
                    f'--burst_reply_size {self.burst_reply_size} '
                    f'--duration {self.args.duration} '
                    f'--client_csv_file {burst_client_file} '
                    f'{"--disable_pcap" if self.args.disable_pcap else ""} '
                    f'> {self.exp_dir}/burst_client_{client_host.name}_out.log 2>&1 &'
                )
                proc = client_host.popen(burst_client_cmd, shell=True)
                self.processes.append(proc)
                
        # Wait for the experiment to finish
        logger.info(f"Waiting for TCP traffic experiment to complete (duration: {self.args.duration}s)...")
        time.sleep(self.args.duration + 2)
        logger.info("TCP traffic experiment completed")

    def run_experiment(self):
        """Run the complete experiment."""
        dataset = None
        try:
            # Setup and start network
            self.setup_experiment()
            self.start_network()
            
            # Configure the SimpleDeflection control plane
            logger.info("Sending BEE packets for SimpleDeflection control plane")
            for switch in self.topology.get_leaf_switches():
                logger.info(f"Sending BEE packets to switch {switch}")
                self.control_plane.send_bee_packets(switch)
            
            # Run queue logger for debugging
            for i, switch in enumerate(self.topology.get_leaf_switches()):
                queue_logger_proc = subprocess.Popen(
                    f"python3 queue_logger.py --port 909{i} --log {self.exp_dir}/queue_log_{switch}.txt",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.processes.append(queue_logger_proc)
            
            # Run CLI if requested
            if (self.args.cli):
                logger.info("Starting Mininet CLI for debugging")
                self.topology.net.start_net_cli()
            else:
            # Run collection experiment - get receiver logs but don't generate dataset yet
                self.run_collection()
            
            # Wait for all processes to finish
            logger.info("Waiting for all processes to complete...")
            for proc in self.processes:
                try:
                    proc.wait(timeout=5)  # Wait for each process to finish
                except subprocess.TimeoutExpired:
                    logger.warning("Process timeout - terminating")
                    proc.terminate()
            
            # Kill queue logger
            for i, switch in enumerate(self.topology.get_leaf_switches()):
                queue_logger_proc = subprocess.Popen(
                    f"pkill -f queue_logger.py --port 909{i}",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.processes.append(queue_logger_proc) 

            time.sleep(5)

            datalib.process_and_merge_all_data(self.topology, exp_dir)

        except Exception as e:
            logger.error(f"Error in experiment: {e}") 
            traceback.print_exc()
            raise
        finally:
            self.stop_network()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run SimpleDeflection packet collection experiment')
    
    # Basic experiment parameters
    parser.add_argument('--duration', '-d', type=int, default=5, 
                        help='Duration of the experiment in seconds (default: 30)')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (default: timestamp)')
    
    # Network configuration - only leaf-spine parameters since we're fixed to SimpleDeflection
    parser.add_argument('--n_hosts', type=int, default=4, 
                        help='Number of hosts (default: 4)')
    parser.add_argument('--n_leaf', type=int, default=2, 
                        help='Number of leaf switches (default: 2)')
    parser.add_argument('--n_spine', type=int, default=2, 
                        help='Number of spine switches (default: 2)')
    parser.add_argument('--bw', type=int, default=10, 
                        help='Link bandwidth in Mbps (default: 10)')
    parser.add_argument('--delay', type=float, default=0, 
                        help='Link delay in ms (default: 0)')
    parser.add_argument('--queue_rate', type=int, default=1000,
                        help='Queue rate in Mbps (default: 100)')
    parser.add_argument('--queue_depth', type=int, default=64,
                        help='Queue depth in packets (default: 64)')

    # Collection parameters
    parser.add_argument('--n_clients', type=int, default=1,
                        help='Number of clients (default: 1)')
    parser.add_argument('--n_servers', type=int, default=1,
                        help='Number of servers (default: 1)')
    parser.add_argument('--flow_iat', type=float, default=0.1, 
                        help='Background Inter-arrival time between consecutive flows in seconds (default: 0.1)')
    parser.add_argument('--congestion_control', type=str, default='cubic', # not happening yet since it's udp
                        help='Congestion control algorithm (default: cubic)')
    parser.add_argument('--flow_size', type=int, default=1000,
                        help='Flow size in bytes (default: 1000)')
    parser.add_argument('--bursty_reply_size', type=int, default=4000,
                        help='Bursty reply size in bytes (default: 4000)')
    parser.add_argument('--burst_interval', type=float, default=0.2,
                        help='Bursty interval in seconds (default: 0.2)')
    parser.add_argument('--burst_servers', type=int, default=1,
                        help='Number of servers to use for bursty traffic (default: 1)')
    parser.add_argument('--burst_clients', type=int, default=None,
                    help='Number of clients running bursty traffic (default: half of total clients)')

    # Debug options
    parser.add_argument('--cli', action='store_true', 
                        help='Start Mininet CLI for debugging')
    # parser.add_argument('--host_pcap', action='store_true', 
    #                     help='Enable packet capture on hosts')
    parser.add_argument('--disable_pcap', action='store_true', 
                        help='Disable packet capture on hosts')
    parser.add_argument('--switch_pcap', action='store_true', 
                        help='Enable packet capture on switches')
    parser.add_argument('--disable_metrics', action='store_true', 
                        help='Disable metrics collection')
    parser.add_argument('--policy', type=str, choices=['simple_deflection', 'ecmp', 'dist_preemptive_deflection', 'quantile_preemptive_deflection', 'rl_deflection'], 
                    default='simple_deflection', help='P4 program to use (default: simple_deflection)')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    runner = CollectionRunner(args)
    dataset = runner.run_experiment()
    logger.info("Experiment completed successfully")
    return dataset # return for the collection batch script

if __name__ == "__main__":
    main()