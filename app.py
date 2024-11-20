import argparse
import socket
import time
import random
from abc import ABC, abstractmethod
from client import BaseClient, BurstyClient, BackgroundClient, IperfClient
from server import BaseServer, BurstyServer, BackgroundServer, IperfServer
import logging
import os
import csv
import configparser
import sqlite3
import pandas as pd
import numpy as np


class App(ABC):
    def __init__(self, mode, log=True):
        self.mode = mode
        self.client = None
        self.server = None
        self.config = self.load_config('config.ini')
        if log:
            self.setup_logging()

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def setup_logging(log_file='tmp/app.log'):
        os.makedirs('tmp', exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()  # This will also print logs to console
            ])

    @staticmethod
    def load_config(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    def cleanup(self):
        if self.server:
            self.server.stop()
        # remove log files


class BurstyApp(App):
    def __init__(self, args):
        mode = args.mode
        host_ip = args.host_ip
        server_ips = args.server_ips
        reply_size = args.reply_size
        incast_scale = args.incast_scale
        qps = args.qps
        super().__init__(mode)
        if mode == 'server':
            self.server = BurstyServer(reply_size=reply_size, ip=args.host_ip, exp_id=args.exp_id)
        elif mode == 'client':
            if server_ips is None:
                raise ValueError("server_ips must be provided for client mode")
            self.client = BurstyClient(server_ips, reply_size, qps=qps, incast_scale=incast_scale, exp_id=args.exp_id)
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'server':
            self.server.start()
        elif self.mode == 'client':
            self.client.start()

class BackgroundApp(App):
    def __init__(self, args):
        super().__init__(args.mode)
        mode = args.mode
        # host_id = args.host_id
        host_ip = args.host_ip
        server_ips = args.server_ips
        flow_ids = args.flow_ids
        flow_sizes = args.flow_sizes
        inter_arrival_times = args.iat
        congestion_control = self.config['background']['cong_proto']
        print("CONG CONTROL", congestion_control)

        if mode == 'server':
            self.server = BackgroundServer(ip=host_ip, exp_id=args.exp_id)
        elif mode == 'client':
            # Pass necessary flow data for the client
            self.client = BackgroundClient(
                server_ips,
                flow_ids,
                flow_sizes,
                inter_arrival_times,
                congestion_control=congestion_control,
                exp_id=args.exp_id
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
        mode = args.mode
        server_ip = args.server_ips
        # packet_size = args.packet_size
        super().__init__(mode)
        if mode == 'server':
            self.server = BaseServer(port=12345, ip=args.host_ip)  # Use a basic server listener
        elif mode == 'client':
            if server_ip is None:
                raise ValueError("server_ip must be provided for client mode")
            self.client = BaseClient(server_ip[0])
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'server':
            self.server.start()  # Blocking call to start listening for packets
        elif self.mode == 'client':
            self.client.start()  # Send a single packet and then exit

class IperfApp(App):
    def __init__(self, args):
        mode = args.mode
        server_ip = args.server_ips
        super().__init__(mode)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--type', choices=['bursty', 'background', 'single'], required=True)
    parser.add_argument('--host_ip', type=str, required=False, help="host IP address (required for servers)")
    # parser.add_argument('--host_id', type=str, required=False, help="sender id: required only for a sender")
    parser.add_argument('--server_ips', required=False, nargs='+', help="List of server IPs (bursty app)")
    parser.add_argument('--reply_size', required=False, type=int, default=40000)
    parser.add_argument('--flow_ids',  required=False, nargs='+', type=int, help="List of flow IDs (background app)")
    parser.add_argument('--flow_sizes', required=False, nargs='+', type=int, help="List of flow sizes (background app)")
    parser.add_argument('--iat', required=False, nargs='+', type=float, help="List of inter-arrival times (background app)")
    parser.add_argument('--incast_scale', required=False, type=int, default=5, help="Number of servers to send requests to in a single query (bursty app)")
    parser.add_argument('--qps', required=False, type=int, default=4000, help="Queries per second (bursty app)")
    parser.add_argument('--exp_id', required=True, type=str, help="Experiment ID")
    args = parser.parse_args()

    # TODO input validation
    if args.type == 'bursty':
        app = BurstyApp(args)
    elif args.type == 'background':
        app = BackgroundApp(args)
    elif args.type == 'single':
        app = SimplePacketApp(args)
    else:
        raise ValueError("Invalid application type. Choose 'bursty' or 'background'.")
    app.run()
    # app.collect_and_write_metrics(f'tmp/metrics_{args.host_id}.csv')

if __name__ == "__main__":
    main()