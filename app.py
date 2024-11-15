import argparse
import socket
import time
import random
from abc import ABC, abstractmethod
from client import BurstyClient, BackgroundClient
from server import BurstyServer, BackgroundServer
import logging
import os
import csv
import configparser
import sqlite3
import pandas as pd
import numpy as np
from metrics import FlowMetricsManager

flowtracker = FlowMetricsManager()

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
        server_ips = args.server_ips
        reply_size = args.reply_size
        super().__init__(mode)
        if mode == 'server':
            self.server = BurstyServer(reply_size)
        elif mode == 'client':
            if server_ips is None:
                raise ValueError("server_ips must be provided for client mode")
            self.client = BurstyClient(server_ips, reply_size)
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'server':
            self.server.start()
        elif self.mode == 'client':
            self.client.run()

class BackgroundApp(App):
    def __init__(self, args):
        super().__init__(args.mode)
        mode = args.mode
        # host_id = args.host_id
        server_ips = args.server_ips
        flow_ids = args.flow_ids
        flow_sizes = args.flow_sizes
        inter_arrival_times = args.iat
        congestion_control = self.config['background']['cong_proto']
        print("CONG CONTROL", congestion_control)

        if mode == 'server':
            self.server = BackgroundServer()
        elif mode == 'client':
            # Pass necessary flow data for the client
            self.client = BackgroundClient(
                server_ips,
                flow_ids,
                flow_sizes,
                inter_arrival_times,
                congestion_control=congestion_control
            )
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'client':
            self.client.run()
        elif self.mode == 'server':
            self.server.start()

    def run(self):
        if self.mode == 'client':
            self.client.run()
        elif self.mode == 'server':
            self.server.start()  # Blocking call, usually you'd use threading for multiple servers

    # def collect_and_write_metrics(self, output_file):
    #     """Aggregate metrics from all servers and write to a CSV file."""
    #     all_metrics = []

    #     # Collect metrics from each server
    #     for server in self.servers:
    #         all_metrics.extend(server.get_metrics())

    #     # Write aggregated metrics to a CSV file
    #     with open(output_file, mode='w', newline='') as file:
    #         writer = csv.writer(file)
    #         writer.writerow(["server_id", "client_id", "bytes_received", "flow_completion_time", "throughput_mbps"])
    #         for metric in all_metrics:
    #             writer.writerow([
    #                 metric["server_id"],
    #                 metric["client_id"],
    #                 metric["bytes_received"],
    #                 metric["flow_completion_time"],
    #                 metric["throughput_mbps"]
    #             ])
    #     logging.info(f"All metrics written to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--type', choices=['bursty', 'background'], required=True)
    # parser.add_argument('--host_id', type=str, required=False, help="sender id: required only for a sender")
    parser.add_argument('--server_ips', required=False, nargs='+', help="List of server IPs (bursty app)")
    parser.add_argument('--reply_size', required=False, type=int, default=40000)
    parser.add_argument('--flow_ids',  required=False, nargs='+', type=int, help="List of flow IDs (background app)")
    parser.add_argument('--flow_sizes', required=False, nargs='+', type=int, help="List of flow sizes (background app)")
    parser.add_argument('--iat', required=False, nargs='+', type=float, help="List of inter-arrival times (background app)")
    args = parser.parse_args()

    # TODO input validation
    if args.type == 'bursty':
        app = BurstyApp(args)
    elif args.type == 'background':
        app = BackgroundApp(args)
    else:
        raise ValueError("Invalid application type. Choose 'bursty' or 'background'.")
    app.run()
    # app.collect_and_write_metrics(f'tmp/metrics_{args.host_id}.csv')

if __name__ == "__main__":
    main()