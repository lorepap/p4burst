import argparse
import socket
import time
import random
from abc import ABC, abstractmethod
from client import BaseClient, BurstyClient, BackgroundClient, IperfClient, DataCollectionClient, TestCollectionClient
from server import BaseServer, BurstyServer, BackgroundServer, IperfServer, DataCollectionServer, TestCollectionServer
import logging
import os
import csv
import configparser
import sqlite3
import pandas as pd
import numpy as np


class App(ABC):
    def __init__(self, args):
        self.mode = args.mode
        self.client = None
        self.server = None
        self.config = self.load_config('config.ini')
        if not args.disable_logging:
            if not os.path.exists(f'tmp/{args.exp_id}'):
                os.makedirs(f'tmp/{args.exp_id}')
            self.setup_logging(log_file=f'tmp/{args.exp_id}/app.log')

    @abstractmethod
    def run(self):
        pass

    # @staticmethod
    # def setup_logging(log_file='tmp/app.log'):
    #     print('setup logging...')
    #     os.makedirs('tmp', exist_ok=True)
    #     logging.basicConfig(level=logging.DEBUG, 
    #                     format='%(asctime)s - %(levelname)s - %(message)s',
    #                     handlers=[
    #             logging.FileHandler(log_file, mode='a'),
    #             logging.StreamHandler()  # This will also print logs to console
    #         ])

    @staticmethod
    def setup_logging(log_file='tmp/app.log'):
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Create a stream (console) handler that logs only errors (ERROR and above)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Get the root logger and configure it
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Overall logger level
        # Remove any pre-existing handlers
        logger.handlers = []
        # Add our handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    @staticmethod
    def load_config(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    def cleanup(self):
        if self.server:
            self.server.stop()


class BurstyApp(App):
    def __init__(self, args):
        super().__init__(args)
        mode = args.mode
        server_ips = args.server_ips
        reply_size = self.config.getint('bursty', 'reply_size')
        qps = self.config.getint('bursty', 'qps')
        if mode == 'server':
            self.server = BurstyServer(reply_size=reply_size, ip=args.host_ip, exp_id=args.exp_id)
        elif mode == 'client':
            if server_ips is None:
                raise ValueError("server_ips must be provided for client mode")
            self.client = BurstyClient(server_ips, reply_size, duration=args.duration, qps=qps, exp_id=args.exp_id)
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'server':
            self.server.start()
        elif self.mode == 'client':
            self.client.start()

class BackgroundApp(App):
    def __init__(self, args):
        super().__init__(args)
        mode = args.mode
        # host_id = args.host_id
        host_ip = args.host_ip
        server_ips = args.server_ips
        flow_ids = args.flow_ids
        flow_sizes = args.flow_sizes
        inter_arrival_times = args.iat
        congestion_control = args.congestion_control

        if mode == 'server':
            self.server = BackgroundServer(ip=host_ip, exp_id=args.exp_id)
        elif mode == 'client':
            # Pass necessary flow data for the client
            self.client = BackgroundClient(
                server_ips,
                flow_ids,
                flow_sizes,
                inter_arrival_times,
                duration=args.duration,
                exp_id=args.exp_id,
                congestion_control=congestion_control
            )
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'client':
            self.client.start()
        elif self.mode == 'server':
            self.server.start()

class SimplePacketApp(App):
    def __init__(self, args):
        super().__init__(args)
        mode = args.mode
        server_ip = args.server_ips
        # packet_size = args.packet_size
        if mode == 'server':
            self.server = BaseServer(port=12345, ip=args.host_ip, exp_id=args.exp_id)  # Use a basic server listener
        elif mode == 'client':
            if server_ip is None:
                raise ValueError("server_ip must be provided for client mode")
            self.client = BaseClient(server_ip=server_ip[0])
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'server':
            self.server.start()  # Blocking call to start listening for packets
        elif self.mode == 'client':
            self.client.start()  # Send a single packet and then exit

class IperfApp(App):
    def __init__(self, args):
        super().__init__(args)
        mode = args.mode
        server_ip = args.server_ips
        if mode == 'server':
            self.server = IperfServer(port=12345, ip=args.host_ip)  # Use a basic server listener
        elif mode == 'client':
            if server_ip is None:
                raise ValueError("server_ip must be provided for client mode")
            self.client = IperfClient(server_ip[0], duration=10)
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'server':
            self.server.start()  # Blocking call to start listening for packets
        elif self.mode == 'client':
            self.client.start()  # Send a single packet and then exit


class TestCollectionApp(App):
    """
    N clients to 1 server
    """
    def __init__(self, args):
        super().__init__(args)
        if self.mode == 'server':
            self.server = TestCollectionServer(ip=args.server_ips[0], port=args.port, exp_id=args.exp_id, log_file=args.server_csv_file)
        elif self.mode == 'client':
            self.client = TestCollectionClient(server_ip=args.server_ips[0], server_port=args.port, 
                            num_packets=args.num_packets, interval=args.interval, num_flows=args.num_flows,
                            exp_id=args.exp_id, congestion_control=args.congestion_control, packet_size=args.packet_size)

    def run(self):
        if self.mode == 'server':
            self.server.start()
        elif self.mode == 'client':
            self.client.start()

class DataCollectionApp(App):
    """
    Client-server app that simulates mixed background and bursty traffic.
    """
    def __init__(self, args):
        super().__init__(args)
        if self.mode == 'server':
            self.server = DataCollectionServer(
                ip=args.server_ips[0], 
                port=args.port, 
                exp_id=args.exp_id, 
                log_file=args.server_csv_file,
                burst_reply_size=args.burst_reply_size
            )
        elif self.mode == 'client':
            self.client = DataCollectionClient(
                server_ips=args.server_ips, 
                server_port=args.port,
                num_packets=args.num_packets, 
                interval=args.interval, 
                #num_flows=args.num_flows,
                exp_id=args.exp_id, 
                congestion_control=args.congestion_control, 
                packet_size=args.packet_size,
                burst_interval=args.burst_interval,
                burst_servers=args.burst_servers,
                burst_reply_size=args.burst_reply_size
            )

    def run(self):
        if self.mode == 'server':
            self.server.start()
        elif self.mode == 'client':
            self.client.start()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--disable_logging', action='store_true', help="Disable logging")
    parser.add_argument('--exp_id', required=False, type=str, help="Experiment ID")
    parser.add_argument('--type', choices=['bursty', 'background', 'single', 'collect'], required=True,
                      help="Type of application to run")
    parser.add_argument('--host_ip', type=str, required=False, help="Host IP address")
    parser.add_argument('--server_ips', required=False, nargs='+', help="List of server IPs")
    parser.add_argument('--duration', type=int, help="Duration for client (in seconds)")
    parser.add_argument('--congestion_control', type=str, default='cubic', help="Congestion control algorithm (for background app)")
    parser.add_argument('--port', type=int, default=12345, help="Port (for mixed app)")

    # Create a group for each application type's arguments
    bursty_group = parser.add_argument_group('bursty', 'Arguments for bursty application')
    bursty_group.add_argument('--reply_size', required=False, type=int, default=40000)
    bursty_group.add_argument('--incast_scale', required=False, type=int, help="Number of servers to send requests in a single query")
    bursty_group.add_argument('--qps', required=False, type=int, default=4000, help="Queries per second (bursty app)")

    background_group = parser.add_argument_group('background', 'Arguments for background application')
    background_group.add_argument('--flow_ids', required=False, nargs='+', type=int, help="List of flow IDs")
    background_group.add_argument('--flow_sizes', required=False, nargs='+', type=int, help="List of flow sizes")
    background_group.add_argument('--iat', required=False, nargs='+', type=float, help="List of inter-arrival times")

    single_group = parser.add_argument_group('single', 'Arguments for simple packet application')
    # No additional arguments needed for single group

    # collect_group = parser.add_argument_group('test_collect', 'Arguments for data collection application')
    # collect_group.add_argument('--num_packets', type=int, default=1000, help="Number of packets per flow")
    # collect_group.add_argument('--interval', type=float, default=0.001, help="Interval between packets")
    # collect_group.add_argument('--num_flows', type=int, default=1, help="Number of flows (for collect app)")
    # collect_group.add_argument('--server_csv_file', type=str, help="Server CSV file (for collect app)")
    # collect_group.add_argument('--packet_size', type=int, default=1000, help="Packet size (for collect app)")

    mixed_group = parser.add_argument_group('collect', 'Arguments for mixed traffic collection')
    mixed_group.add_argument('--num_packets', type=int, default=1000, help="Number of packets per background flow")
    mixed_group.add_argument('--interval', type=float, default=0.001, help="Interval between packets")
    mixed_group.add_argument('--num_flows', type=int, default=1, help="Number of background flows")
    mixed_group.add_argument('--server_csv_file', type=str, help="Server CSV file")
    mixed_group.add_argument('--packet_size', type=int, default=1000, help="Packet size")
    mixed_group.add_argument('--burst_interval', type=float, default=1.0, help="Time between bursts (seconds)")
    mixed_group.add_argument('--burst_servers', type=int, default=2, help="Number of servers in each burst")
    mixed_group.add_argument('--burst_reply_size', type=int, default=40000, help="Size of burst response")


    args = parser.parse_args()

    # Rest of your code remains the same
    if args.type == 'bursty':
        app = BurstyApp(args)
    elif args.type == 'background':
        app = BackgroundApp(args)
    elif args.type == 'single':
        app = SimplePacketApp(args)
    elif args.type == 'test_collect':
        app = TestCollectionApp(args)
    elif args.type == 'collect':
        app = DataCollectionApp(args)
    else:
        raise ValueError("Invalid application type. Choose 'bursty' or 'background'.")
    app.run()
    # app.collect_and_write_metrics(f'tmp/metrics_{args.host_id}.csv')

if __name__ == "__main__":
    main()