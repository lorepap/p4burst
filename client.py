import os
import socket
import time
import random
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import concurrent.futures
import statistics
import traceback
import sqlite3
import pandas as pd
import threading
import numpy as np
import traceback
from metrics import FlowMetricsManager


class BaseClient(ABC):
    def __init__(self):
        self.ip = self.get_host_ip()

    @abstractmethod
    def send_request(self, server_ip):
        pass

    def run(self):
        while True:
            for server_ip in self.server_ips:
                self.send_request(server_ip)
            time.sleep(random.uniform(1, 5)) 

    # TODO: move this to a base host class (for both server and client)
    def get_host_ip(self):
        try:
            # Create a socket to determine the outgoing IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))  # Doesn't actually send any data
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            raise ValueError("Could not determine host IP address")


class BurstyClient(BaseClient):
    def __init__(self, server_ips, reply_size=40000):
        super().__init__()
        self.server_ips = server_ips # only burst app needs a subset of servers
        self.reply_size = reply_size
        self.fct_stats = defaultdict(list)
        self.qct_stats = []

    def send_request(self, server_ip):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                start_time = time.time()
                s.connect((server_ip, 12345))
                logging.info(f"[{self.ip}]: Sending request to {server_ip}:12345")
                s.sendall(b'Request')
                data = b''
                while len(data) < self.reply_size:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                end_time = time.time()
                fct = end_time - start_time
                self.fct_stats[server_ip].append(fct)
                logging.info(f"[{self.ip}]: Received {len(data)} bytes from {server_ip}:12345. FCT: {fct:.6f} seconds")
                return fct
            except Exception as e:
                logging.error(f"[{self.ip}]: Error connecting to {server_ip}: {e}")
                return None

    def send_query(self):
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.server_ips)) as executor: # parallel requests
            future_to_server = {executor.submit(self.send_request, server_ip): server_ip for server_ip in self.server_ips}
            for future in concurrent.futures.as_completed(future_to_server):
                server_ip = future_to_server[future]
                try:
                    fct = future.result()
                    if fct is not None:
                        logging.info(f"[{self.ip}]: Completed request to {server_ip}")
                except Exception as e:
                    logging.error(f"[{self.ip}]: Request to {server_ip} generated an exception: {e}")
                    logging.error(traceback.format_exc())
        end_time = time.time()
        qct = end_time - start_time
        self.qct_stats.append(qct)
        logging.info(f"[{self.ip}]: Query completed. QCT: {qct:.6f} seconds")

    def run(self):
        while True:
            self.send_query()
            time.sleep(random.uniform(1, 5))

    def print_stats(self):
        logging.info(f"[{self.ip}]: FCT stats:")
        all_fcts = []
        for server_ip, fcts in self.fct_stats.items():
            if fcts:
                avg_fct = statistics.mean(fcts)
                min_fct = min(fcts)
                max_fct = max(fcts)
                all_fcts.extend(fcts)
                logging.info(f"  {server_ip}: min={min_fct:.6f}s, max={max_fct:.6f}s, avg={avg_fct:.6f}s")
            else:
                logging.info(f"  {server_ip}: No successful requests")
        
        if all_fcts:
            overall_avg_fct = statistics.mean(all_fcts)
            logging.info(f"[{self.ip}]: Overall average FCT: {overall_avg_fct:.6f}s")
        else:
            logging.info(f"[{self.ip}]: No successful requests to any server")

        if self.qct_stats:
            avg_qct = statistics.mean(self.qct_stats)
            min_qct = min(self.qct_stats)
            max_qct = max(self.qct_stats)
            logging.info(f"[{self.ip}]: QCT stats: min={min_qct:.6f}s, max={max_qct:.6f}s, avg={avg_qct:.6f}s")
        else:
            logging.info(f"[{self.ip}]: No queries completed")


class BackgroundClient(BaseClient):
    def __init__(self, server_ips, flow_ids, flow_sizes, inter_arrival_times, congestion_control='cubic'):
        super().__init__()
        self.congestion_control = congestion_control
        self.server_ips = server_ips
        self.flow_ids = flow_ids
        self.flow_sizes = flow_sizes
        self.inter_arrival_times = inter_arrival_times

    def send_request(self, flow_id, server_ip, flow_size):
        """Send a single flow (request) of data to a server and wait for acknowledgment."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Set congestion control and connect
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
                logging.info(f"Connecting to {server_ip}:12345")
                s.connect((server_ip, 12345))
                # Prepare the data payload with the flow ID prefixed
                flow_id_prefix = f"{int(flow_id):08d}".encode('utf-8')
                data = flow_id_prefix + (b'x' * flow_size)
                print('getting into the flowtracker')
                FlowMetricsManager().start_flow(flow_id, self.ip, server_ip, flow_size)
                logging.info(f"Sending flow {flow_id} to {server_ip}:12345")
                s.sendall(data)  # Send all data in one go

            except Exception as e:
                logging.error(f"[{self.ip}]: Error sending traffic to {server_ip}: {e}")
                traceback.print_exc()

    def run(self):
        """Send flows to each server based on inter-arrival times and flow sizes."""
        for flow_id, inter_arrival_time, flow_size, server_ip in zip(
            self.flow_ids, self.inter_arrival_times, self.flow_sizes, self.server_ips
        ):
            self.send_request(flow_id, server_ip, flow_size)
            time.sleep(float(inter_arrival_time))