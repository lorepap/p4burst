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
import argparse
from datetime import datetime
import os
import time
import random
import traceback
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

# Mapping for P4 programs
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

# Globals
exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")

def setup_logging(exp_id, disable_logs=False):
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if disable_logs:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.WARNING)
        console.setFormatter(fmt)
        root.setLevel(logging.WARNING)
        root.addHandler(console)
    else:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(fmt)
        os.makedirs(f'tmp/{exp_id}', exist_ok=True)
        flog = os.path.join('tmp', exp_id, 'collection_runner.log')
        file_handler = logging.FileHandler(flog)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(fmt)
        root.setLevel(logging.DEBUG)
        root.addHandler(console)
        root.addHandler(file_handler)
    return logging.getLogger('CollectionRunner')

class CollectionRunner:
    """
    Runner specifically for collection experiments that analyze packet reordering.
    Uses various TCP/UDP workloads under a P4 control plane.
    """
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.exp_id = args.exp_id or exp_id
        self.exp_dir = f'tmp/{self.exp_id}'
        os.makedirs(self.exp_dir, exist_ok=True)

        # Topology and control plane parameters
        self.policy = args.policy
        self.n_hosts = args.n_hosts
        self.n_leaf = args.n_leaf
        self.n_spine = args.n_spine
        self.bw = args.bw
        self.delay = args.delay
        self.queue_rate = args.queue_rate
        self.queue_depth = args.queue_depth
        self.logaritmic_deflecting_margin = args.logaritmic_deflecting_margin
        self.alpha = args.alpha
        self.m_prio_num_entries = args.m_prio_num_entries
        self.m_prio_rank_entries = args.m_prio_rank_entries

        # Traffic parameters
        self.n_clients = args.n_clients
        self.n_servers = args.n_servers
        self.flow_iat = args.flow_iat
        self.flow_size = args.flow_size
        self.burst_interval = args.burst_interval
        self.burst_servers = args.burst_servers
        self.burst_clients = args.burst_clients or max(1, self.n_clients // 2)
        self.burst_reply_size = args.bursty_reply_size
        self.duration = args.duration
        self.bg_port = args.bg_port
        self.burst_port = args.burst_port
        self.disable_pcap = args.disable_pcap
        self.switch_pcap = args.switch_pcap
        self.disable_metrics = args.disable_metrics
        self.cli = args.cli

        self.processes = []
        self.logger.info(f"Initialized runner with exp_id={self.exp_id}")

    def setup_experiment(self):
        self.logger.info("Setting up topology and control plane...")
        self.topology = LeafSpineTopology(
            self.n_hosts, self.n_leaf, self.n_spine,
            self.bw, self.delay, p4_program_paths[self.policy]
        )
        cp_cls = p4_control_plane[self.policy]
        cp_args = dict(queue_rate=self.queue_rate,
                       queue_depth=self.queue_depth,
                       burst_port=self.burst_port,
                       bg_port=self.bg_port)
        if 'preemptive' in self.policy:
            cp_args.update(
                alpha=self.alpha,
                m_prio_num_entries=self.m_prio_num_entries,
                m_prio_rank_entries=self.m_prio_rank_entries
            )
        self.control_plane = cp_cls(self.topology, **cp_args)
        self.logger.info("Experiment setup complete")

    def start_network(self):
        self.logger.info("Starting network...")
        self.topology.generate_topology()
        if self.switch_pcap:
            self.topology.enable_switch_pcap()
        self.topology.start_network()
        self.control_plane.generate_control_plane()
        self.topology.net.program_switches()
        self.logger.info("Network started")

    def stop_network(self):
        self.logger.info("Stopping network...")
        if hasattr(self, 'topology') and self.topology.net:
            self.topology.net.stopNetwork()
        self.logger.info("Network stopped")

    def run_collection(self):
        # Implementation of mixed background/bursty traffic
        # Similar to previous canvas: start servers, then clients
        pass  # omitted for brevity

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
            # Run collection or CLI
            if self.cli:
                self.logger.info("Launching Mininet CLI")
                self.topology.net.start_net_cli()
            else:
                self.run_collection()
            # Wait and gather metrics
            for p in self.processes:
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.terminate()
            time.sleep(5)
            _, fct_avg = calculate_fct(self.exp_dir, self.exp_dir)
            _, qct_avg = calculate_qct(self.exp_dir, self.exp_dir)
            self.logger.info(f"Avg FCT: {fct_avg:.5f}, Avg QCT: {qct_avg:.5f}")
        except Exception as e:
            self.logger.error(f"Experiment error: {e}")
            traceback.print_exc()
            raise
        finally:
            self.stop_network()
        return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run SimpleDeflection packet collection experiment")
    parser.add_argument('--disable_logs', action='store_true', help='Suppress INFO logs')
    parser.add_argument('--exp_id', type=str, default=None, help='Experiment ID')
    parser.add_argument('--duration', type=int, default=30, help='Experiment duration (s)')
    parser.add_argument('--n_hosts', type=int, default=4)
    parser.add_argument('--n_leaf', type=int, default=2)
    parser.add_argument('--n_spine', type=int, default=2)
    parser.add_argument('--bw', type=int, default=10)
    parser.add_argument('--delay', type=float, default=0)
    parser.add_argument('--queue_rate', type=int, default=1000)
    parser.add_argument('--queue_depth', type=int, default=64)
    parser.add_argument('--policy', type=str, choices=list(p4_program_paths), default='simple_deflection')

    parser.add_argument('--n_clients', type=int, default=1)
    parser.add_argument('--n_servers', type=int, default=1)
    parser.add_argument('--flow_iat', type=float, default=0.1)
    parser.add_argument('--flow_size', type=int, default=1000)
    parser.add_argument('--bursty_reply_size', type=int, default=4000)
    parser.add_argument('--burst_interval', type=float, default=0.2)
    parser.add_argument('--burst_servers', type=int, default=1)
    parser.add_argument('--burst_clients', type=int, default=None)

    parser.add_argument('--bg_port', type=int, default=12345)
    parser.add_argument('--burst_port', type=int, default=12346)
    parser.add_argument('--disable_pcap', action='store_true')
    parser.add_argument('--switch_pcap', action='store_true')
    parser.add_argument('--disable_metrics', action='store_true')
    parser.add_argument('--cli', action='store_true')

    parser.add_argument('--logaritmic_deflecting_margin', type=int, default=1)
    parser.add_argument('--m_prio_num_entries', type=int, default=8)
    parser.add_argument('--m_prio_rank_entries', type=int, default=8)
    parser.add_argument('--alpha', type=float, default=0.5)

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.exp_id or exp_id, disable_logs=args.disable_logs)
    logger.info("Collection Runner starting")
    runner = CollectionRunner(args, logger)
    runner.run_experiment()
    logger.info("Experiment completed successfully")

if __name__ == '__main__':
    main()
