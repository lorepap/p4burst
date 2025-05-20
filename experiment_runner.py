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
import csv

from topology import LeafSpineTopology
from control_plane import (
    ECMPControlPlane,
    RLDeflectionControlPlane,
    SimpleDeflectionControlPlane,
    DistPreemptiveDeflectionControlPlane,
    QuantilePreemptiveDeflectionControlPlane
)
from utils.config_override import update_p4_consts
from utils.stats import calculate_fct, calculate_qct
from p4utils.utils.sswitch_thrift_API import SimpleSwitchThriftAPI

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
        
        for server_host in bg_servers:
            if not self.disable_logging:
                out = f" > {self.exp_dir}/server_bg_{server_host.name}.log 2>&1 &"
            else:
                out = " &"
            cmd = (
                f"python3 server.py --exp_id {self.exp_id} "
                f"--host_ip {server_host.IP()} --port {self.bg_port} "
                f"--reply_size {self.flow_size} "
                f"{'--disable_pcap' if self.disable_pcap else ''} "
                f"{'--disable_logging' if self.disable_logging else ''} "
                f"{out}"
            )
            server_host.cmd(cmd)
        for server_host in bursty_servers:
            if not self.disable_logging:
                out = f" > {self.exp_dir}/server_burst_{server_host.name}.log 2>&1 &"
            else:
                out = " &"
            cmd = (
                f"python3 server.py --exp_id {self.exp_id} "
                f"--host_ip {server_host.IP()} --port {self.burst_port} "
                f"--reply_size {self.burst_reply_size} "
                f"{'--disable_pcap' if self.disable_pcap else ''} "
                f"{'--disable_logging' if self.disable_logging else ''} "
                f"{out}"
            )
            server_host.cmd(cmd)
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
        
        for client_host in clients:
            if not self.disable_logging:
                out = f" > {self.exp_dir}/client_bg_{client_host.name}.log 2>&1 &"
            else:
                out = " &"
            cmd_bg = (
                f"python3 bg_client.py --exp_id {self.exp_id} "
                f"--server_ips {bg_server_ips} --port {self.bg_port} "
                f"--reply_size {self.flow_size} --interval {self.flow_iat} "
                f"--duration {self.duration} "
                f"{'--disable_pcap' if self.disable_pcap else ''} "
                f"{'--disable_logging' if self.disable_logging else ''} "
                f"{out}"
            )
            client_host.cmd(cmd_bg)

        for client_host in bursty_clients:
            if not self.disable_logging:
                out = f" > {self.exp_dir}/client_burst_{client_host.name}.log 2>&1 &"
            else:
                out = " &"
            cmd_burst = (
                f"python3 bursty_client.py --exp_id {self.exp_id} "
                f"--server_ips {bursty_server_ips} --port {self.burst_port} "
                f"--reply_size {self.burst_reply_size} --interval {self.burst_interval} "
                f"--burst_servers {self.burst_servers} --duration {self.duration} "
                f"{'--disable_pcap' if self.disable_pcap else ''} "
                f"{'--disable_logging' if self.disable_logging else ''} "
                f"{out}"
            )
            client_host.cmd(cmd_burst)
        # Wait for the experiment to finish
        self.logger.info(f"Waiting for TCP traffic experiment to complete (duration: {self.duration}s)...")
        time.sleep(self.duration + 2)
        self.logger.info("TCP traffic experiment completed")

    def collect_counters(self):
        """Collect packet counters from each switch and write to CSV."""
        self.logger.info("Collecting packet counters from switches...")
        results = {}
        csv_file = os.path.join(self.exp_dir, "switch_counters.csv")
        
        total_reg_sum = 0
        total_ing_reg_sum = 0
        total_egr_reg_sum = 0
        total_egress_packets = 0
        total_ingress_packets = 0  # Per i pacchetti che completano l'ingress
        total_implicit_drops = 0  # Nuovo contatore per il totale dei pacchetti droppati implicitamente
        max_process_time = 0  # Per tenere traccia del tempo massimo globale
        max_ing_process_time = 0
        max_egr_process_time = 0
        
        for switch in self.topology.get_leaf_switches():
            results[switch] = {}
            switch_id = int(switch[1:]) - 1
            thrift_port = 9090 + switch_id
            self.logger.info(f"⇒ Connecting to {switch} on Thrift port {thrift_port}")

            api = SimpleSwitchThriftAPI(thrift_port=thrift_port)
            
            # Try a test read to force an exception in case of failure
            try:
                test = api.counter_read("packet_counter", 0)
            except Exception as e:
                self.logger.error(f"Failed test read from {switch}: {e}")
                results[switch]['total'] = -1
                results[switch]['ingress_total'] = -1  # Nuovo campo per conteggio pacchetti ingress
                results[switch]['deflection'] = -1
                results[switch]['drop'] = -1
                results[switch]['implicit_drop'] = -1  # Nuovo campo per il contatore
                results[switch]['egress'] = -1
                results[switch]['reg_sum'] = -1
                results[switch]['ing_reg_sum'] = -1
                results[switch]['egr_reg_sum'] = -1
                results[switch]['avg_traversal'] = -1
                results[switch]['max_traversal'] = -1
                results[switch]['avg_ing_traversal'] = -1
                results[switch]['max_ing_traversal'] = -1
                results[switch]['avg_egr_traversal'] = -1
                results[switch]['max_egr_traversal'] = -1
                continue

            # Read the actual counters
            packet_count = api.counter_read("packet_counter", 0)
            if packet_count is None:
                self.logger.error(f"{switch}: counter_read('packet_counter') returned None")
                results[switch]['total'] = -1
            else:
                results[switch]['total'] = packet_count[1]  # Get the count value
                
            # Leggi il contatore ingress_packet_counter
            ingress_packet_count = api.counter_read("ingress_packet_counter", 0)
            if ingress_packet_count is None:
                self.logger.error(f"{switch}: counter_read('ingress_packet_counter') returned None")
                results[switch]['ingress_total'] = -1
            else:
                results[switch]['ingress_total'] = ingress_packet_count[1]
                total_ingress_packets += ingress_packet_count[1]  # Aggiungi al totale

            deflection_count = api.counter_read("deflect_counter", 0)
            if deflection_count is None:
                self.logger.error(f"{switch}: counter_read('deflect_counter') returned None")
                results[switch]['deflection'] = -1
            else:
                results[switch]['deflection'] = deflection_count[1]  # Get the count value

            drop_count = api.counter_read("drop_counter", 0)
            if drop_count is None:
                self.logger.error(f"{switch}: counter_read('drop_counter') returned None")
                results[switch]['drop'] = -1
            else:
                results[switch]['drop'] = drop_count[1]  # Get the count value
                
            # Leggi il contatore implicitely_dropped
            implicit_drop_count = api.counter_read("implicitely_dropped", 0)
            if implicit_drop_count is None:
                self.logger.error(f"{switch}: counter_read('implicitely_dropped') returned None")
                results[switch]['implicit_drop'] = -1
            else:
                results[switch]['implicit_drop'] = implicit_drop_count[1]
                total_implicit_drops += implicit_drop_count[1]  # Aggiungi al totale
                
            # Read egress packet counter
            egress_packet_count = api.counter_read("egress_packet_counter", 0)
            if egress_packet_count is None:
                self.logger.error(f"{switch}: counter_read('egress_packet_counter') returned None")
                results[switch]['egress'] = -1
                results[switch]['avg_traversal'] = -1
            else:
                results[switch]['egress'] = egress_packet_count[1]  # Get the count value
                total_egress_packets += egress_packet_count[1]
                
            # Read reg_sum register (total processing time)
            reg_sum = api.register_read("reg_sum", 0)
            if reg_sum is None:
                self.logger.error(f"{switch}: register_read('reg_sum') returned None")
                results[switch]['reg_sum'] = -1
                results[switch]['avg_traversal'] = -1
            else:
                results[switch]['reg_sum'] = reg_sum
                total_reg_sum += reg_sum
                
                # Calculate average traversal time per switch
                if results[switch]['egress'] > 0:
                    results[switch]['avg_traversal'] = reg_sum / results[switch]['egress']
                else:
                    results[switch]['avg_traversal'] = -1
                
            # Read ingress processing time registers
            ing_reg_sum = api.register_read("reg_ing_sum", 0)
            if ing_reg_sum is None:
                self.logger.error(f"{switch}: register_read('reg_ing_sum') returned None")
                results[switch]['ing_reg_sum'] = -1
                results[switch]['avg_ing_traversal'] = -1
            else:
                results[switch]['ing_reg_sum'] = ing_reg_sum
                total_ing_reg_sum += ing_reg_sum
                
                # Calculate average ingress traversal time per switch using the new counter
                if results[switch]['ingress_total'] > 0:
                    results[switch]['avg_ing_traversal'] = ing_reg_sum / results[switch]['ingress_total']
                else:
                    results[switch]['avg_ing_traversal'] = -1
            
            # Read egress processing time registers
            egr_reg_sum = api.register_read("reg_egr_sum", 0)
            if egr_reg_sum is None:
                self.logger.error(f"{switch}: register_read('reg_egr_sum') returned None")
                results[switch]['egr_reg_sum'] = -1
                results[switch]['avg_egr_traversal'] = -1
            else:
                results[switch]['egr_reg_sum'] = egr_reg_sum
                total_egr_reg_sum += egr_reg_sum
                
                # Calculate average egress traversal time per switch
                if results[switch]['egress'] > 0:
                    results[switch]['avg_egr_traversal'] = egr_reg_sum / results[switch]['egress']
                else:
                    results[switch]['avg_egr_traversal'] = -1
        
            # Leggi il registro del tempo massimo di processamento totale
            reg_max_time = api.register_read("reg_max_time", 0)
            if reg_max_time is None:
                self.logger.error(f"{switch}: register_read('reg_max_time') returned None")
                results[switch]['max_traversal'] = -1
            else:
                results[switch]['max_traversal'] = reg_max_time
                # Aggiorna il massimo globale se necessario
                if reg_max_time > max_process_time:
                    max_process_time = reg_max_time
                    
            # Leggi il registro del tempo massimo di processamento ingress
            ing_max_time = api.register_read("reg_ing_max_time", 0)
            if ing_max_time is None:
                self.logger.error(f"{switch}: register_read('reg_ing_max_time') returned None")
                results[switch]['max_ing_traversal'] = -1
            else:
                results[switch]['max_ing_traversal'] = ing_max_time
                # Aggiorna il massimo globale se necessario
                if ing_max_time > max_ing_process_time:
                    max_ing_process_time = ing_max_time
                    
            # Leggi il registro del tempo massimo di processamento egress
            egr_max_time = api.register_read("reg_egr_max_time", 0)
            if egr_max_time is None:
                self.logger.error(f"{switch}: register_read('reg_egr_max_time') returned None")
                results[switch]['max_egr_traversal'] = -1
            else:
                results[switch]['max_egr_traversal'] = egr_max_time
                # Aggiorna il massimo globale se necessario
                if egr_max_time > max_egr_process_time:
                    max_egr_process_time = egr_max_time

        # Calculate average traversal time if we have valid data
        avg_traversal_time = 0
        avg_ing_traversal_time = 0
        avg_egr_traversal_time = 0
        
        if total_egress_packets > 0:
            avg_traversal_time = total_reg_sum / total_egress_packets
            avg_egr_traversal_time = total_egr_reg_sum / total_egress_packets
            
            self.logger.info(f"Global average packet traversal time: {avg_traversal_time} ns")
            self.logger.info(f"Global average egress traversal time: {avg_egr_traversal_time} ns")
        
        # Calcola il tempo medio di ingress usando il nuovo contatore
        if total_ingress_packets > 0:
            avg_ing_traversal_time = total_ing_reg_sum / total_ingress_packets
            self.logger.info(f"Global average ingress traversal time: {avg_ing_traversal_time} ns")
            
        self.logger.info(f"Global maximum packet traversal time: {max_process_time} ns")
        self.logger.info(f"Global maximum ingress traversal time: {max_ing_process_time} ns")
        self.logger.info(f"Global maximum egress traversal time: {max_egr_process_time} ns")
        self.logger.info(f"Total implicitly dropped packets: {total_implicit_drops}")

        # Write results to CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['switch_id', 'total', 'ingress_total', 'deflected', 'dropped', 'implicitly_dropped', 'egress', 
                            'reg_sum', 'ing_reg_sum', 'egr_reg_sum', 
                            'avg_traversal', 'avg_ing_traversal', 'avg_egr_traversal',
                            'max_traversal', 'max_ing_traversal', 'max_egr_traversal'])
            for switch, counts in results.items():
                writer.writerow([
                    switch, 
                    counts['total'],
                    counts['ingress_total'],  # Nuovo campo
                    counts['deflection'],
                    counts['drop'],
                    counts['implicit_drop'],
                    counts['egress'],
                    counts['reg_sum'],
                    counts['ing_reg_sum'],
                    counts['egr_reg_sum'],
                    counts['avg_traversal'],
                    counts['avg_ing_traversal'],
                    counts['avg_egr_traversal'],
                    counts['max_traversal'],
                    counts['max_ing_traversal'],
                    counts['max_egr_traversal']
                ])
        
        # Write summary data to a separate file
        summary_file = os.path.join(self.exp_dir, "traversal_summary.csv")
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['total_egress_packets', 'total_ingress_packets', 'total_reg_sum', 'total_ing_reg_sum', 'total_egr_reg_sum',
                            'global_avg_traversal_time_ns', 'global_avg_ing_traversal_time_ns', 'global_avg_egr_traversal_time_ns',
                            'global_max_traversal_time_ns', 'global_max_ing_traversal_time_ns', 'global_max_egr_traversal_time_ns',
                            'total_implicit_drops'])
            writer.writerow([total_egress_packets,
                            total_ingress_packets,  # Nuovo campo
                            total_reg_sum, total_ing_reg_sum, total_egr_reg_sum,
                            avg_traversal_time, avg_ing_traversal_time, avg_egr_traversal_time,
                            max_process_time, max_ing_process_time, max_egr_process_time, 
                            total_implicit_drops])
        
        self.logger.info(f"Counter data written to {csv_file}")
        self.logger.info(f"Traversal summary written to {summary_file}")
        return results

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

            time.sleep(5)
            
            # Collect packet counters before stopping the network
            self.collect_counters()
            
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

