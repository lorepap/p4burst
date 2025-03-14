#!/usr/bin/env python3

"""
Collection Runner - Simplified runner for packet collection experiments

This script runs packet collection experiments using UDP flows with custom headers
for reordering detection and analysis, specifically using the SimpleDeflection control plane.
"""

import sys
import logging

# Set up root logger first thing
def setup_logging():
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
            logging.FileHandler("collection_runner.log")  # Add file logging too
        ]
    )
    
    # Return a logger for this module
    return logging.getLogger("CollectionRunner")

# Setup logging early
logger = setup_logging()

# Rest of your imports
import os
import time
import random
import traceback
import argparse
import subprocess
from datetime import datetime

from topology import LeafSpineTopology, DumbbellTopology
from control_plane import SimpleDeflectionControlPlane
from metrics import FlowMetricsManager
from utils.data_handling import collect_switch_logs, combine_datasets

# Add a sanity check log to verify logging is working
logger.info("Collection Runner starting - logging is active")

class CollectionRunner:
    """
    Runner specifically for collection experiments that analyze packet reordering.
    Uses SimpleDeflection control plane.
    """
    def __init__(self, args):
        self.args = args
        self.exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.processes = []
        
        # Create experiment directory
        self.exp_dir = f'tmp/{self.exp_id}'
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Configure experiment parameters - fixed to simple deflection
        self.topology_type = 'leafspine'  # Only using leaf-spine topology
        self.n_hosts = args.n_hosts
        self.n_leaf = args.n_leaf
        self.n_spine = args.n_spine
        self.bw = args.bw
        self.delay = args.delay
        self.p4_program = 'sd/sd.p4'  # Fixed P4 program for simple deflection
        self.queue_rate = args.queue_rate
        self.queue_depth = args.queue_depth
        
        # Collection parameters
        self.num_flows = args.num_flows
        self.num_packets = args.num_packets
        self.interval = args.interval
        self.congestion_control = args.congestion_control
        
        # Initialize flow metrics if tracking enabled
        if not args.disable_metrics:
            self.flow_metrics = FlowMetricsManager(self.exp_id)
            
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
            self.p4_program
        )
        
        # Create control plane - fixed to SimpleDeflection
        self.control_plane = SimpleDeflectionControlPlane(self.topology, queue_rate=self.queue_rate, 
                                                          queue_depth=self.queue_depth)
        
        logger.info("Experiment setup complete")

    def start_network(self):
        """Start the network and generate the control plane."""
        logger.info("Starting network...")
        self.topology.generate_topology()
        
        # Enable packet captures if requested
        if self.args.switch_pcap:
            self.topology.enable_switch_pcap()
        if self.args.host_pcap:
            self.enable_pcap_hosts()
            
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

    def enable_pcap_hosts(self):
        """Enable packet capture on host interfaces."""
        pcap_dir = f'{self.exp_dir}/pcap'
        os.makedirs(pcap_dir, exist_ok=True)

        for host in self.topology.net.net.hosts:
            for intf in host.intfList():
                if intf.name != 'lo':
                    cmd = f'tcpdump -i {intf} -w {pcap_dir}/{intf}.pcap &'
                    host.cmd(cmd)
        logger.info("Host packet capture enabled")

    def run_collection(self):
        """Run the collection experiment."""
        logger.info("Starting collection experiment...")
        receiver_logs = []
        
        # Select client and server hosts
        client_host = self.topology.net.net.get('h1')
        server_host = self.topology.net.net.get('h2')
        
        # Prepare log file path
        server_csv_file = f"{self.exp_dir}/receiver_log.csv"
        
        # Start the server
        logger.info(f"Starting collection server on {server_host.name} ({server_host.IP()})...")
        server_cmd = (
            f'python3 -m app --mode server --type collect '
            f'--port 12345 '
            f'--server_ips {server_host.IP()} '
            f'--exp_id {self.exp_id} '
            f'--server_csv_file {server_csv_file} > {self.exp_dir}/server_out.log 2>&1 &'
        )
        server_host.cmd(server_cmd)
        # Give server time to initialize
        time.sleep(1)
        
        # Start the client
        logger.info(f"Starting collection client on {client_host.name} ({client_host.IP()})...")
        client_cmd = (
            f'python3 -m app --mode client '
            f'--type collect '
            f'--server_ips {server_host.IP()} '
            f'--num_packets {self.num_packets} '
            f'--interval {self.interval} '
            f'--num_flows {self.num_flows} '
            f'--congestion_control {self.congestion_control} '
            f'--exp_id {self.exp_id}'
        )
        
        proc = client_host.popen(client_cmd, shell=True, stderr=sys.stderr, stdout=subprocess.DEVNULL)
        self.processes.append(proc)
        
        # Wait for client to finish
        logger.info(f"Waiting for collection to complete (duration: {self.args.duration}s)...")
        time.sleep(self.args.duration + 2)  # Add a small buffer
        
        # Add the receiver log to the list
        receiver_logs.append(server_csv_file)
        logger.info("Collection experiment completed")
        
        # Return the receiver logs for dataset generation later
        return receiver_logs

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
                self.control_plane.send_bee_packets(switch='s1')
            
            # Run queue logger for debugging
            queue_logger_proc = subprocess.Popen(
                f"python3 queue_logger.py --log {self.exp_dir}/queue_log.txt",
                shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.processes.append(queue_logger_proc)
            
            # Run CLI if requested
            if self.args.cli:
                logger.info("Starting Mininet CLI for debugging")
                self.topology.net.start_net_cli()
            
            # Run collection experiment - get receiver logs but don't generate dataset yet
            receiver_logs = self.run_collection()
            
            # Wait for all processes to finish
            logger.info("Waiting for all processes to complete...")
            for proc in self.processes:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Process timeout - terminating")
                    proc.terminate()
            
            # Kill queue logger
            queue_logger_proc.terminate()
            
            # Now generate the dataset after all processes have finished but before stopping the network
            logger.info("Generating final dataset...")
            switch_datasets = collect_switch_logs(self.topology, self.exp_dir)
            dataset = combine_datasets(switch_datasets, receiver_logs, self.exp_dir)
            logger.info(f"Final dataset created at {self.exp_dir}/combined_dataset.csv")
            
            return dataset
            
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
    parser.add_argument('--exp_id', type=str, default=None, 
                        help='Experiment ID (default: timestamp)')
    parser.add_argument('--duration', '-d', type=int, default=30, 
                        help='Duration of the experiment in seconds (default: 30)')
    
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
    parser.add_argument('--queue_depth', type=int, default=100,
                        help='Queue depth in packets (default: 100)')
    
    # Collection parameters
    parser.add_argument('--num_flows', type=int, default=1, 
                        help='Number of flows to generate (default: 1)')
    parser.add_argument('--num_packets', type=int, default=1000, 
                        help='Number of packets per flow (default: 1000)')
    parser.add_argument('--interval', type=float, default=0.001, 
                        help='Inter-packet interval in seconds (default: 0.001)')
    parser.add_argument('--congestion_control', type=str, default='cubic', 
                        help='Congestion control algorithm (default: cubic)')
    
    # Debug options
    parser.add_argument('--cli', action='store_true', 
                        help='Start Mininet CLI for debugging')
    parser.add_argument('--host_pcap', action='store_true', 
                        help='Enable packet capture on hosts')
    parser.add_argument('--switch_pcap', action='store_true', 
                        help='Enable packet capture on switches')
    parser.add_argument('--disable_metrics', action='store_true', 
                        help='Disable metrics collection')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    runner = CollectionRunner(args)
    dataset = runner.run_experiment()
    logger.info("Experiment completed successfully")
    return dataset

if __name__ == "__main__":
    main()