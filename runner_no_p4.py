#!/usr/bin/env python3

import argparse
import os
import time
import random
import traceback
from mininet.net import Mininet
from mininet.node import Host, OVSSwitch, Controller
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info

def get_args():
    parser = argparse.ArgumentParser(description='Run network experiment')
    parser.add_argument('--bw', '-b', type=int, help='Bandwidth in Mbps', default=10)
    parser.add_argument('--latency', '-d', type=float, help='Latency in ms', default=10)
    parser.add_argument('--duration', type=int, default=2, help='Duration of the experiment in seconds')
    parser.add_argument('--cli', action='store_true', help='Enable Mininet CLI')
    return parser.parse_args()

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.net = None

    def setup_network(self):
        # Create the Mininet object
        self.net = Mininet(controller=Controller, link=TCLink, switch=OVSSwitch)

        self.net.addController('c0')

        # Create hosts
        h1 = self.net.addHost('h1')
        h2 = self.net.addHost('h2')
        h3 = self.net.addHost('h3')
        h4 = self.net.addHost('h4')

        # Create switches (non-P4)
        s1 = self.net.addSwitch('s1')
        s2 = self.net.addSwitch('s2')

        # Add links between hosts and switches
        self.net.addLink(h1, s1, bw=self.args.bw, delay=f'{self.args.latency}ms', max_queue_size=1000)
        self.net.addLink(h2, s1, bw=self.args.bw, delay=f'{self.args.latency}ms', max_queue_size=1000)
        self.net.addLink(h3, s2, bw=self.args.bw, delay=f'{self.args.latency}ms', max_queue_size=1000)
        self.net.addLink(h4, s2, bw=self.args.bw, delay=f'{self.args.latency}ms', max_queue_size=1000)

        # Add link between switches
        self.net.addLink(s1, s2, bw=self.args.bw, delay=f'{self.args.latency}ms')

        # Enable pcap capture on all interfaces
        # Create directory for pcaps if it doesn't exist
        pcap_dir = 'pcap'
        if not os.path.exists(pcap_dir):
            os.makedirs(pcap_dir)

        # Enable pcap on switches
        # for switch in [s1, s2]:
        #     for intf in switch.intfList():
        #         if intf.name != 'lo':
        #             cmd = f'tcpdump -i {intf} -w {pcap_dir}/{intf}.pcap &'
        #             switch.cmd(cmd)

        for host in [h1, h2, h3, h4]:
            for intf in host.intfList():
                if intf.name != 'lo':
                    cmd = f'tcpdump -i {intf} -w {pcap_dir}/{intf}.pcap &'
                    host.cmd(cmd)

    def start_network(self):
        self.net.start()
        # If CLI is enabled, start it
        if self.args.cli:
            CLI(self.net)

    def stop_network(self):
        if self.net:
            self.net.stop()

    def run_app(self):
        # Get hosts
        h1 = self.net.get('h1')
        h3 = self.net.get('h3')

        # Change congestion control
        h1.cmd('sysctl -w net.ipv4.tcp_congestion_control=bbr')

        h1.cmd('ifconfig h1-eth0 txqueuelen 1000')

        # Start server on h3
        print(f"Starting server on h3 ({h3.IP()})...")
        # h3.cmd(f'python3 -m app --mode server --host_ip {h3.IP()} --type single &')
        h3.cmd(f'iperf3 -s &')

        time.sleep(2)  # Give the server some time to start up

        # Start client on h1
        print(f"Starting client on h1 ({h1.IP()})...")
        # h1.cmd(f'python3 -m app --mode client --type single --server_ips {h3.IP()} &')
        h1.cmd(f'iperf3 -c {h3.IP()} -t {self.args.duration + 10}&')

    def run_experiment(self):
        try:
            self.setup_network()
            self.start_network()
            self.run_app()
            print(f"Experiment running for {self.args.duration} seconds...")
            time.sleep(self.args.duration)
        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            self.stop_network()

def main():
    setLogLevel('info')
    args = get_args()
    experiment = ExperimentRunner(args)
    experiment.run_experiment()

if __name__ == "__main__":
    main()
