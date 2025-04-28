#!/usr/bin/env python3

import sys
import logging
from datetime import datetime
import os
import time
import random
import traceback
import argparse
import subprocess

from topology import LeafSpineTopology, DumbbellTopology
from control_plane import (
    ECMPControlPlane,
    RLDeflectionControlPlane,
    SimpleDeflectionControlPlane,
    DistPreemptiveDeflectionControlPlane,
    QuantilePreemptiveDeflectionControlPlane
)
from utils.config_override import update_p4_consts
from utils.stats import calculate_fct, calculate_qct

# Set up root logger first thing
def setup_logging(exp_id, exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
    os.chmod(exp_dir, 0o777) 
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join('tmp', exp_id, "collection_runner.log"))
        ]
    )
    return logging.getLogger("CollectionRunner")


p4_program_paths = {
    'simple_deflection': 'Simple_Deflection/sd.p4',
    'ecmp': 'ecmp.p4',
    'dist_preemptive_deflection': 'Dist_PD/distpd.p4',
    'quantile_preemptive_deflection': 'Quantile_PD/quantilepd.p4',
    'rl_deflection': 'evaluation/evaluation.p4'
}

p4_const_paths = {
    'simple_deflection': 'p4src/Simple_Deflection/includes/sd_consts.p4',
    'ecmp': 'p4src/ecmp.p4',
    'dist_preemptive_deflection': 'p4src/Dist_PD/includes/distpd_consts.p4',
    'quantile_preemptive_deflection': 'p4src/Quantile_PD/includes/quantilepd_consts.p4',
    'rl_deflection': 'p4src/evaluation/includes/evaluation_consts.p4'
}

p4_control_plane = {
    'simple_deflection': SimpleDeflectionControlPlane,
    'rl_deflection': RLDeflectionControlPlane,
    'ecmp': ECMPControlPlane,
    'dist_preemptive_deflection': DistPreemptiveDeflectionControlPlane,
    'quantile_preemptive_deflection': QuantilePreemptiveDeflectionControlPlane
}


class CollectionRunner:
    """
    Runner specifically for collection experiments that analyze packet reordering.
    Uses SimpleDeflection control plane.
    """
    def __init__(self, args):
        self.args = args
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
        self.duration = args.duration
        
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
        self.burst_port = args.burst_port
        self.bg_port = args.bg_port
        self.logaritmic_deflecting_margin = args.logaritmic_deflecting_margin
        self.m_prio_num_entries = args.m_prio_num_entries
        self.m_prio_rank_entries = args.m_prio_rank_entries
        self.alpha = args.alpha
        self.disable_pcap = args.disable_pcap
        self.disable_logging = args.disable_logging
        
        self.exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f'tmp/{self.exp_id}'

        # setup_logging si occupa di mkdir+chmod
        self.logger = setup_logging(self.exp_id, self.exp_dir)
        
        # Initialize flow metrics if tracking enabled
        # if not args.disable_metrics:
        #     self.flow_metrics = FlowMetricsManager(self.exp_id)
            
        self.logger.info(f"Initialized collection runner with experiment ID: {self.exp_id}")

    def setup_experiment(self):
        """Set up the experiment topology with SimpleDeflection control plane."""
        self.logger.info(f"Setting up leaf-spine topology with SimpleDeflection control plane")
        
        # Create topology - fixed to leaf-spine for simple deflection
        self.topology = LeafSpineTopology(
            self.n_hosts, 
            self.n_leaf, 
            self.n_spine, 
            self.bw, 
            self.delay,
            p4_program_paths[self.policy]
        )
        if self.policy == 'dist_preemptive_deflection':
            self.control_plane = p4_control_plane[self.policy](self.topology, queue_rate=self.queue_rate,
                                                          queue_depth=self.queue_depth, burst_port=self.burst_port, bg_port=self.bg_port, alpha=self.alpha,
                                                          m_prio_num_entries=self.m_prio_num_entries, m_newm_num_entries=self.m_prio_num_entries,
                                                          m_prio_rank_entries=self.m_prio_rank_entries, m_newm_rank_entries=self.m_prio_rank_entries)
        else:    
            self.control_plane = p4_control_plane[self.policy](self.topology, queue_rate=self.queue_rate, 
                                                          queue_depth=self.queue_depth, burst_port=self.burst_port, bg_port=self.bg_port)
        
        self.logger.info("Experiment setup complete")

    def start_network(self):
        """Start the network and generate the control plane."""
        self.logger.info("Starting network...")
        self.topology.generate_topology()
        
        # Enable packet captures if requested
        if self.args.switch_pcap:
            self.topology.enable_switch_pcap()
            
        self.topology.start_network()
        self.control_plane.generate_control_plane()
        self.topology.net.program_switches()  # insert the rules
        self.logger.info("Network started")

    def stop_network(self):
        """Stop the network."""
        if hasattr(self, 'topology') and self.topology.net:
            self.logger.info("Stopping network...")
            self.topology.net.stopNetwork()
            self.logger.info("Network stopped")

    def run_collection(self):
        """Run the traffic collection experiment with separate background and bursty traffic."""
        self.logger.info("Starting TCP traffic experiment...")
        receiver_logs = []
        
        # Use available hosts as servers
        hosts = self.topology.net.net.hosts
        # Random servers
        bg_servers = random.sample(hosts, self.n_servers)
        bursty_servers = random.sample(hosts, self.burst_servers)
        # Random clients
        clients = random.sample(hosts, self.n_clients)
        ncores = os.cpu_count() or 1
        current_core = 0
        
        # Start all servers - both background and burst
        for server_host in bg_servers:
            self.logger.info(f"Starting background TCP server on {server_host.name} ({server_host.IP()})...")
            selceted_core = current_core % ncores
            current_core += 1
            bg_server_cmd = (
                'python3 -m app --mode server '
                f'--exp_id {self.exp_id} '
                '--traffic_type background '
                f'--port {self.bg_port} '
                f'--host_ip {server_host.IP()} '
                f'--core {selceted_core} '
                f'--server_ips {server_host.IP()} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""} '
                f'{"--disable_logging" if self.disable_logging else ""} '
                f'> {self.exp_dir}/bg_server_{server_host.name}_out.log 2>&1 &'
            )
            server_host.cmd(bg_server_cmd)
            
        for server_host in bursty_servers:
            # Burst TCP server
            self.logger.info(f"Starting burst TCP server on {server_host.name} ({server_host.IP()})...")
            selceted_core = current_core % ncores
            current_core += 1
            burst_server_cmd = (
                'python3 -m app --mode server '
                f'--exp_id {self.exp_id} '
                '--traffic_type burst '
                f'--port {self.burst_port} '
                f'--core {selceted_core} '
                f'--host_ip {server_host.IP()} '
                f'--server_ips {server_host.IP()} '
                f'--burst_reply_size {self.burst_reply_size} '
                f'{"--disable_pcap" if self.disable_pcap else ""} '
                f'{"--disable_logging" if self.disable_logging else ""} '
                f'> {self.exp_dir}/burst_server_{server_host.name}_out.log 2>&1 &'
            )
            server_host.cmd(burst_server_cmd)
        
        # Give servers time to initialize
        time.sleep(5)

        # Start clients - ALL clients run background traffic, but only a subset runs bursty traffic
        client_csv_files = []
        
        # Determine how many clients will be bursty (random subset)
        # Add a new parameter to control this or use a fixed percentage
        num_bursty_clients = self.burst_clients
        bursty_clients = random.sample(hosts, num_bursty_clients)
        self.logger.info(f"Selected {num_bursty_clients}/{self.n_clients} clients to generate bursty traffic")
        bg_server_ips = ' '.join([server.IP() for server in bg_servers])
        bursty_server_ips = ' '.join([server.IP() for server in bursty_servers])
        print(f"Background server IPs: {bg_server_ips}")
        print(f"Bursty server IPs: {bursty_server_ips}")
        for client_host in clients:
            # Background TCP client - ALL clients run this
            bg_client_file = f"{self.exp_dir}/bg_client_{client_host.name}_log.csv"
            client_csv_files.append(bg_client_file)
            selceted_core = current_core % ncores
            current_core += 1
            self.logger.info(f"Starting background TCP client on {client_host.name} ({client_host.IP()})...")
            
            bg_client_cmd = (
                'python3 -m app '
                '--mode client '
                f'--exp_id {self.exp_id} '
                '--traffic_type background '
                f'--server_ips {bg_server_ips} '
                f'--bg_flow_iat {self.flow_iat} '
                f'--host_ip {client_host.IP()} '
                f'--port {self.bg_port} '
                f'--core {selceted_core} '
                f'--flow_size {self.flow_size} '
                f'--duration {self.duration} '
                #f'--client_csv_file {bg_client_file} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""} '
                f'{"--disable_logging" if self.args.disable_logging else ""} '
                f'> {self.exp_dir}/bg_client_{client_host.name}_out.log 2>&1 &'
            )
            proc = client_host.popen(bg_client_cmd, shell=True)
            self.processes.append(proc)
            
            # Burst TCP client - ONLY a subset of clients run this
        for client_host in bursty_clients:
            burst_client_file = f"{self.exp_dir}/burst_client_{client_host.name}_log.csv"
            client_csv_files.append(burst_client_file)
            selceted_core = current_core % ncores
            current_core += 1
                
            self.logger.info(f"Starting burst TCP client on {client_host.name} ({client_host.IP()})...")
                
            burst_client_cmd = (
                'python3 -m app '
                '--mode client '
                f'--exp_id {self.exp_id} '
                '--traffic_type burst '
                f'--server_ips {bursty_server_ips} '
                f'--port {self.burst_port} '
                f'--host_ip {client_host.IP()} '
                f'--burst_interval {self.burst_interval} '
                f'--burst_servers {self.burst_servers} '
                f'--burst_reply_size {self.burst_reply_size} '
                f'--duration {self.duration} '
                f'--core {selceted_core} '
                #f'--client_csv_file {burst_client_file} '
                f'{"--disable_pcap" if self.disable_pcap else ""} '
                f'{"--disable_logging" if self.disable_logging else ""} '
                f'> {self.exp_dir}/burst_client_{client_host.name}_out.log 2>&1 &'
            )
            proc = client_host.popen(burst_client_cmd, shell=True)
            self.processes.append(proc)
                
        # Wait for the experiment to finish
        self.logger.info(f"Waiting for TCP traffic experiment to complete (duration: {self.duration}s)...")
        time.sleep(self.duration + 2)
        self.logger.info("TCP traffic experiment completed")

    def run_experiment(self):
        """Run the complete experiment."""
        try:
            # Setup and start network
            
            update_p4_consts(
                p4_const_paths[self.policy],
                self.queue_depth,
                self.logaritmic_deflecting_margin,
                self.alpha,
                self.m_prio_num_entries,
                self.m_prio_rank_entries
            )
            self.setup_experiment()
            self.start_network()
            
            # Configure the SimpleDeflection control plane
            self.logger.info("Sending BEE packets for SimpleDeflection control plane")
            # Versione corretta 2: if dentro il ciclo
            for switch in self.topology.get_leaf_switches():
                if self.policy != 'ecmp':
                    self.logger.info(f"Sending BEE packets to switch {switch}")
                    self.control_plane.send_bee_packets(switch)
            
            # Run queue logger for debugging
            for i, switch in enumerate(self.topology.get_leaf_switches()):
                queue_logger_proc = subprocess.Popen(
                    f"python3 queue_logger.py --port 909{i} --log {self.exp_dir}/queue_log_{switch}.txt",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.processes.append(queue_logger_proc)
            
            # Run CLI if requested
            if (self.args.cli):
                self.logger.info("Starting Mininet CLI for debugging")
                self.topology.net.start_net_cli()
            else:
            # Run collection experiment - get receiver logs but don't generate dataset yet
                self.run_collection()
            
            # Wait for all processes to finish
            self.logger.info("Waiting for all processes to complete...")
            time.sleep(self.duration + 10)
            for proc in self.processes:
                try:
                    proc.wait(timeout=5)  # Wait for each process to finish
                except subprocess.TimeoutExpired:
                    self.logger.warning("Process timeout - terminating")
                    proc.terminate()
            
            # Kill queue logger
            for i, switch in enumerate(self.topology.get_leaf_switches()):
                queue_logger_proc = subprocess.Popen(
                    f"pkill -f queue_logger.py --port 909{i}",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.processes.append(queue_logger_proc) 

            time.sleep(5)
            
            _, fct_avg = calculate_fct(self.exp_dir, self.exp_dir)
            _, qct_avg = calculate_qct(self.exp_dir, self.exp_dir)
            self.logger.info(f"Average FCT: {fct_avg:.5f}")
            self.logger.info(f"Average QCT: {qct_avg:.5f}")
            #datalib.process_and_merge_all_data(self.topology, exp_dir)
            self.logger.info("Experiment completed successfully")
        except Exception as e:
            self.logger.error(f"Error in experiment: {e}") 
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
    parser.add_argument('--burst_clients', type=int, default=2,
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
    parser.add_argument('--burst_port', type=int, default=12346)
    parser.add_argument('--bg_port', type=int, default=12345)
    parser.add_argument('--logaritmic_deflecting_margin', type=int, default=1,
                        help='Deflecting margin for preemptive deflections (default: 1)')
    parser.add_argument('--m_prio_num_entries', type=int, default=8,
                        help='Number of entries in the m-prio table (default: 8)')
    parser.add_argument('--m_prio_rank_entries', type=int, default=8,
                        help='Number of entries in the m-prio rank table (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for preemptive deflections (default: 0.5)')
    parser.add_argument('--disable_logging', action='store_true')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    runner = CollectionRunner(args)
    dataset = runner.run_experiment()
    return dataset # return for the collection batch script

if __name__ == "__main__":
    main()
    '''
#!/usr/bin/env python3

import sys
import logging
from datetime import datetime
import os
import time
import random
import traceback
import argparse
import subprocess

from topology import LeafSpineTopology, DumbbellTopology
from control_plane import (
    ECMPControlPlane,
    RLDeflectionControlPlane,
    SimpleDeflectionControlPlane,
    DistPreemptiveDeflectionControlPlane,
    QuantilePreemptiveDeflectionControlPlane
)
from utils.config_override import update_p4_consts
from utils.stats import calculate_fct, calculate_qct

# Set up root logger first thing
def setup_logging(exp_id):
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join('tmp', exp_id, "collection_runner.log"))
        ]
    )
    return logging.getLogger("CollectionRunner")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run mixed TCP traffic collection experiments'
    )
    # App parameters
    parser.add_argument('--config_file', default='config.ini',
                        help='Path to app config file')
    parser.add_argument('--disable_logging', action='store_true',
                        help='Disable logging in the app')
    parser.add_argument('--disable_pcap', action='store_true',
                        help='Disable packet capture in the app')
    # Experiment parameters
    parser.add_argument('--duration', '-d', type=int, default=5,
                        help='Duration of the experiment in seconds (default: 5)')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='Experiment ID (default: timestamp)')
    # Topology configuration
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
                        help='Queue rate in Mbps (default: 1000)')
    parser.add_argument('--queue_depth', type=int, default=64,
                        help='Queue depth in packets (default: 64)')
    # Traffic mix parameters
    parser.add_argument('--n_clients', type=int, default=1,
                        help='Number of clients (default: 1)')
    parser.add_argument('--n_servers', type=int, default=1,
                        help='Number of servers (default: 1)')
    parser.add_argument('--flow_iat', type=float, default=0.1,
                        help='Background inter-arrival time between flows in seconds (default: 0.1)')
    parser.add_argument('--congestion_control', type=str, default='cubic',
                        help='Congestion control algorithm (default: cubic)')
    parser.add_argument('--flow_size', type=int, default=1000,
                        help='Flow size in bytes (default: 1000)')
    parser.add_argument('--burst_reply_size', type=int, default=4000,
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
    parser.add_argument('--switch_pcap', action='store_true',
                        help='Enable packet capture on switches')
    parser.add_argument('--disable_metrics', action='store_true',
                        help='Disable metrics collection')
    parser.add_argument('--policy', type=str,
                        choices=[
                            'simple_deflection',
                            'ecmp',
                            'dist_preemptive_deflection',
                            'quantile_preemptive_deflection',
                            'rl_deflection'
                        ],
                        default='simple_deflection',
                        help='P4 program to use (default: simple_deflection)')
    parser.add_argument('--burst_port', type=int, default=12346,
                        help='Port for burst traffic (default: 12346)')
    parser.add_argument('--bg_port', type=int, default=12345,
                        help='Port for background traffic (default: 12345)')
    parser.add_argument('--logaritmic_deflecting_margin', type=int, default=1,
                        help='Deflecting margin for preemptive deflection (default: 1)')
    parser.add_argument('--m_prio_num_entries', type=int, default=8,
                        help='Number of entries in the m-prio table (default: 8)')
    parser.add_argument('--m_prio_rank_entries', type=int, default=8,
                        help='Number of entries in the m-prio rank table (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha value for preemptive deflections (default: 0.5)')
    parser.add_argument('--disable_logging', action='store_true',
                        help='Disable logging in the runner')
    return parser.parse_args()


# Initialize globals
args = parse_args()
exp_id = args.exp_id or datetime.now().strftime("%Y%m%d_%H%M%S")
exp_dir = f'tmp/{exp_id}'
ios.makedirs(exp_dir, exist_ok=True)
logger = setup_logging(exp_id)
logger.info("Collection Runner starting - logging is active")

# P4 paths
p4_program_paths = {
    'simple_deflection': 'Simple_Deflection/sd.p4',
    'ecmp': 'ecmp.p4',
    'dist_preemptive_deflection': 'Dist_PD/distpd.p4',
    'quantile_preemptive_deflection': 'Quantile_PD/quantilepd.p4',
    'rl_deflection': 'evaluation/evaluation.p4'
}

p4_const_paths = {
    'simple_deflection': 'p4src/Simple_Deflection/includes/sd_consts.p4',
    'ecmp': 'p4src/ecmp.p4',
    'dist_preemptive_deflection': 'p4src/Dist_PD/includes/distpd_consts.p4',
    'quantile_preemptive_deflection': 'p4src/Quantile_PD/includes/quantilepd_consts.p4',
    'rl_deflection': 'p4src/evaluation/includes/evaluation_consts.p4'
}

p4_control_plane = {
    'simple_deflection': SimpleDeflectionControlPlane,
    'rl_deflection': RLDeflectionControlPlane,
    'ecmp': ECMPControlPlane,
    'dist_preemptive_deflection': DistPreemptiveDeflectionControlPlane,
    'quantile_preemptive_deflection': QuantilePreemptiveDeflectionControlPlane
}


class CollectionRunner:
    def __init__(self, args):
        self.args = args
        self.exp_id = exp_id
        self.exp_dir = exp_dir
        self.processes = []
        # Topology
        self.n_hosts = args.n_hosts
        self.n_leaf = args.n_leaf
        self.n_spine = args.n_spine
        self.bw = args.bw
        self.delay = args.delay
        self.policy = args.policy
        self.queue_rate = args.queue_rate
        self.queue_depth = args.queue_depth
        # Traffic
        self.n_clients = args.n_clients
        self.n_servers = args.n_servers
        self.flow_iat = args.flow_iat
        self.flow_size = args.flow_size
        self.congestion_control = args.congestion_control
        self.burst_reply_size = args.burst_reply_size
        self.burst_interval = args.burst_interval
        self.burst_servers = args.burst_servers
        self.burst_clients = args.burst_clients
        self.burst_port = args.burst_port
        self.bg_port = args.bg_port
        # Control plane params
        self.logaritmic_deflecting_margin = args.logaritmic_deflecting_margin
        self.m_prio_num_entries = args.m_prio_num_entries
        self.m_prio_rank_entries = args.m_prio_rank_entries
        self.alpha = args.alpha
        logger.info(f"Initialized collection runner with experiment ID: {self.exp_id}")

    def setup_experiment(self):
        logger.info(f"Setting up leaf-spine topology with {self.policy} control plane")
        self.topology = LeafSpineTopology(
            self.n_hosts, self.n_leaf, self.n_spine, self.bw, self.delay,
            p4_program_paths[self.policy]
        )
        if self.policy == 'dist_preemptive_deflection':
            self.control_plane = p4_control_plane[self.policy](
                self.topology,
                queue_rate=self.queue_rate,
                queue_depth=self.queue_depth,
                burst_port=self.burst_port,
                bg_port=self.bg_port,
                alpha=self.alpha,
                m_prio_num_entries=self.m_prio_num_entries,
                m_prio_rank_entries=self.m_prio_rank_entries
            )
        else:
            self.control_plane = p4_control_plane[self.policy](
                self.topology,
                queue_rate=self.queue_rate,
                queue_depth=self.queue_depth,
                burst_port=self.burst_port,
                bg_port=self.bg_port
            )
        logger.info("Experiment setup complete")

    def start_network(self):
        logger.info("Starting network...")
        self.topology.generate_topology()
        if self.args.switch_pcap:
            self.topology.enable_switch_pcap()
        self.topology.start_network()
        self.control_plane.generate_control_plane()
        self.topology.net.program_switches()
        logger.info("Network started")

    def stop_network(self):
        if hasattr(self, 'topology') and self.topology.net:
            logger.info("Stopping network...")
            self.topology.net.stopNetwork()
            logger.info("Network stopped")

    def run_collection(self):
        logger.info("Starting TCP traffic experiment...")
        receiver_logs = []
        hosts = self.topology.net.net.hosts
        servers = random.sample(hosts, self.n_servers)
        clients = random.sample(hosts, self.n_clients)
        # Start servers
        for server_host in servers:
            logger.info(f"Starting background TCP server on {server_host.name}")
            bg_cmd = (
                'python3 -m app --mode server '
                f'--exp_id {self.exp_id} '
                '--traffic_type background '
                f'--host_ip {server_host.IP()} '
                f'--port {self.bg_port} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""}'
                f'{"--disable_logging" if self.args.disable_logging else ""}'
                f'> {self.exp_dir}/bg_server_{server_host.name}_out.log 2>&1 &'
            )
            server_host.cmd(bg_cmd)
            logger.info(f"Starting burst TCP server on {server_host.name}")
            burst_cmd = (
                'python3 -m app --mode server '
                f'--exp_id {self.exp_id} '
                '--traffic_type burst '
                f'--host_ip {server_host.IP()} '
                f'--port {self.burst_port} '
                f'--burst_reply_size {self.burst_reply_size} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""}'
                f'{"--disable_logging" if self.args.disable_logging else ""}'
                f'> {self.exp_dir}/burst_server_{server_host.name}_out.log 2>&1 &'
            )
            server_host.cmd(burst_cmd)
        time.sleep(2)
        logger.info("Starting clients for background and bursty traffic")
        num_bursty = min(self.burst_clients, self.n_clients) if self.burst_clients else max(1, self.n_clients//2)
        bursty_clients = random.sample(clients, num_bursty)
        for client_host in clients:
            bg_log = f"{self.exp_dir}/bg_client_{client_host.name}_log.csv"
            bg_cmd = (
                'python3 -m app --mode client '
                f'--exp_id {self.exp_id} '
                '--traffic_type background '
                f'--host_ip {client_host.IP()} '
                f'--server_ips {' '.join([s.IP() for s in servers])} '
                f'--bg_flow_iat {self.flow_iat} '
                f'--flow_size {self.flow_size} '
                f'--duration {self.args.duration} '
                f'{"--disable_pcap" if self.args.disable_pcap else ""}'
                f'{"--disable_logging" if self.args.disable_logging else ""}'
                f'--client_csv_file {bg_log} '
                f'> {self.exp_dir}/bg_client_{client_host.name}_out.log 2>&1 &'
            )
            client_host.popen(bg_cmd, shell=True)
            if client_host in bursty_clients:
                burst_log = f"{self.exp_dir}/burst_client_{client_host.name}_log.csv"
                burst_cmd = (
                    'python3 -m app --mode client '
                    f'--exp_id {self.exp_id} '
                    '--traffic_type burst '
                    f'--host_ip {client_host.IP()} '
                    f'--server_ips {' '.join([s.IP() for s in servers])} '
                    f'--burst_interval {self.burst_interval} '
                    f'--burst_servers {self.burst_servers} '
                    f'--burst_reply_size {self.burst_reply_size} '
                    f'--duration {self.args.duration} '
                    f'{"--disable_pcap" if self.args.disable_pcap else ""}'
                    f'{"--disable_logging" if self.args.disable_logging else ""}'
                    f'--client_csv_file {burst_log} '
                    f'> {self.exp_dir}/burst_client_{client_host.name}_out.log 2>&1 &'
                )
                client_host.popen(burst_cmd, shell=True)
        time.sleep(self.args.duration + 2)
        logger.info("TCP traffic experiment completed")

    def run_experiment(self):
        dataset = None
        try:
            update_p4_consts(
                p4_const_paths[self.policy],
                self.queue_depth,
                self.logaritmic_deflecting_margin,
                self.alpha,
                self.m_prio_num_entries,
                self.m_prio_rank_entries
            )
            self.setup_experiment()
            self.start_network()
            logger.info("Sending BEE packets for control plane")
            for switch in self.topology.get_leaf_switches():
                if self.policy != 'ecmp':
                    logger.info(f"Sending BEE to switch {switch}")
                    self.control_plane.send_bee_packets(switch)
            for i, switch in enumerate(self.topology.get_leaf_switches()):
                proc = subprocess.Popen(
                    f"python3 queue_logger.py --port 909{i} --log {self.exp_dir}/queue_log_{switch}.txt",
                    shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self.processes.append(proc)
            if self.args.cli:
                logger.info("Starting Mininet CLI")
                self.topology.net.start_net_cli()
            else:
                self.run_collection()
            logger.info("Waiting for processes to finish")
            for proc in self.processes:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("Process timeout, terminating")
                    proc.terminate()
            for i, switch in enumerate(self.topology.get_leaf_switches()):
                subprocess.Popen(
                    f"pkill -f queue_logger.py --port 909{i}", shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            time.sleep(5)
            _, fct_avg = calculate_fct(self.exp_dir, self.exp_dir)
            _, qct_avg = calculate_qct(self.exp_dir, self.exp_dir)
            logger.info(f"Average FCT: {fct_avg:.5f}")
            logger.info(f"Average QCT: {qct_avg:.5f}")
        except Exception as e:
            logger.error(f"Error in experiment: {e}")
            traceback.print_exc()
            raise
        finally:
            self.stop_network()


def main():
    runner = CollectionRunner(args)
    runner.run_experiment()
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
    '''