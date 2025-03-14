import argparse
import socket
import time
import random
from abc import ABC, abstractmethod
from client import BaseClient, BurstyClient, BackgroundClient, IperfClient, CollectionClient
from server import BaseServer, BurstyServer, BackgroundServer, IperfServer, CollectionServer
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


class DataCollectionApp(App):
    """
    N clients to 1 server
    """
    def __init__(self, args):
        super().__init__(args)
        if self.mode == 'server':
            self.server = CollectionServer(ip=args.server_ips[0], port=args.port, exp_id=args.exp_id, log_file=args.server_csv_file)
        elif self.mode == 'client':
            self.client = CollectionClient(server_ip=args.server_ips[0], server_port=args.port, 
                            num_packets=args.num_packets, interval=args.interval, num_flows=args.num_flows,
                            exp_id=args.exp_id, congestion_control=args.congestion_control)

    def run(self):
        if self.mode == 'server':
            self.server.start()
        elif self.mode == 'client':
            self.client.start()
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--type', choices=['bursty', 'background', 'single', 'collect'], required=True)
    parser.add_argument('--host_ip', type=str, required=False, help="host IP address (required for servers)")
    parser.add_argument('--port', type=int, default=12345, help="Port (for collect app)")
    parser.add_argument('--num_packets', type=int, default=1000, help="Number of packets per flow (for collect app)")
    parser.add_argument('--interval', type=float, default=0.001, help="Interval between packets (for collect app)")
    parser.add_argument('--num_flows', type=int, default=1, help="Number of flows (for collect app)")
    parser.add_argument('--server_csv_file', type=str, help="Server CSV file (for collect app)")
    parser.add_argument('--disable_logging', action='store_true', help="Disable logging")
    parser.add_argument('--server_ips', required=False, nargs='+', help="List of server IPs (bursty app)")
    parser.add_argument('--reply_size', required=False, type=int, default=40000)
    parser.add_argument('--flow_ids',  required=False, nargs='+', type=int, help="List of flow IDs (background app)")
    parser.add_argument('--flow_sizes', required=False, nargs='+', type=int, help="List of flow sizes (background app)")
    parser.add_argument('--iat', required=False, nargs='+', type=float, help="List of inter-arrival times (background app)")
    parser.add_argument('--incast_scale', required=False, type=int, help="Number of servers to send requests to in a single query (bursty app)")
    parser.add_argument('--qps', required=False, type=int, default=4000, help="Queries per second (bursty app)")
    parser.add_argument('--exp_id', required=False, type=str, help="Experiment ID")
    parser.add_argument('--duration', required=False, type=int, help="Duration for client")
    parser.add_argument('--congestion_control', required=False, type=str, default='cubic', help="Congestion control algorithm")
    args = parser.parse_args()

    # TODO input validation
    if args.type == 'bursty':
        app = BurstyApp(args)
    elif args.type == 'background':
        app = BackgroundApp(args)
    elif args.type == 'single':
        app = SimplePacketApp(args)
    elif args.type == 'collect':
        app = DataCollectionApp(args)
    else:
        raise ValueError("Invalid application type. Choose 'bursty' or 'background'.")
    app.run()
    # app.collect_and_write_metrics(f'tmp/metrics_{args.host_id}.csv')

if __name__ == "__main__":
    main()