import os
import sys
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
import subprocess

class BaseClient(ABC):
    def __init__(self, congestion_control='cubic', exp_id='', server_ip=None):
        self.ip = self.get_host_ip()
        self.server_ip = server_ip
        self.congestion_control = congestion_control
        self.exp_id = exp_id
        if self.exp_id:
            self.flowtracker = FlowMetricsManager(self.exp_id)

    def send_request(self, server_ip=None, packet_size=1024):
        """
        Sends a single packet to the server.
        
        Args:
            server_ip (str, optional): Server IP to send the packet to. Defaults to self.server_ip.
        """
        if not self.server_ip:
            self.server_ip = server_ip
        if self.server_ip is None:
            raise ValueError("Server IP must be specified")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Connect to the server
                logging.info("Connecting to server...")
                s.connect((server_ip, 12345))
                logging.info(f"[{self.ip}]: Sending single packet to {server_ip}:12345")

                # Prepare the data packet with a specified size
                data = b'P' * packet_size  # Placeholder byte array for the packet
                logging.info("Sending packet...")
                s.sendall(data)  # Send all data in one go

                # Read the response from the server
                response = s.recv(1024)
                logging.info(f"Received response: {response}")

                logging.info(f"[{self.ip}]: Sent {len(data)} bytes to {server_ip}:12345")
            except Exception as e:
                logging.error(f"[{self.ip}]: Error sending single packet to {server_ip}: {e}")
                traceback.print_exc()

    def start(self):
        logging.info("Starting client...")
        self.send_request(self.server_ip, 1024)

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
            logging.warning ("Warning: Could not determine host IP address - setting to None")


class BurstyClient(BaseClient):
    """
    TODO Should integrate the web bursty traffic to generate realistic queries?
    qps: queries per second
    incast_scale: number of servers to send requests to in a single query
    """
    def __init__(self, server_ips, reply_size, duration, qps, congestion_control='cubic', exp_id=''):
        super().__init__(congestion_control, exp_id)
        self.server_ips = server_ips
        self.reply_size = reply_size
        self.qps = qps
        #self.incast_scale = len(server_ips)
        self.fct_stats = defaultdict(list)
        self.qct_stats = []
        self.duration = duration
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(server_ips))

    def send_request(self, server_ip, flow_id):
        """Send a request to the server and wait for the response."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.settimeout(5)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
                s.connect((server_ip, 12346))
                # Send the flow ID as part of the data
                flow_id_prefix = f"{int(flow_id):08d}".encode('utf-8')
                s.sendall(flow_id_prefix + b'x')  # Include flow ID and some data
                # self.flowtracker.start_flow(flow_id, self.ip, self.server_ip, len(flow_id_prefix) + 1)
                # Receive the full response
                data = b''
                while len(data) < int(self.reply_size):
                    try:
                        chunk = s.recv(4096)
                    except socket.timeout:
                        logging.error(f"[{self.ip}]: Timeout connecting to {server_ip}")
                        break
                    if not chunk:
                        break
                    data += chunk
                # Mark the flow as complete
                self.flowtracker.complete_flow(flow_id)
            except Exception as e:
                logging.error(f"[{self.ip}]: Error connecting to {server_ip}: {e}")
                traceback.print_exc()
                return None
            return 1
        
    def send_query(self):
        """Simulate an incast event by sending requests to multiple servers."""
        # Generate unique flow IDs for this query's flows
        flow_ids = {server_ip: random.randint(10000000, 99999999) for server_ip in self.server_ips}
        query_id = random.randint(0, 99999999)
        # Select servers for the incast event
        selected_servers = self.server_ips

        # Schedule all requests concurrently in the persistent executor.
        futures = {
            self.executor.submit(self.send_request, server_ip, flow_ids[server_ip]): server_ip
            for server_ip in selected_servers
        }

        
        # with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_servers)) as executor:
        #     future_to_server = {
        #         executor.submit(self.send_request, server_ip, flow_ids[server_ip]): server_ip
        #         for server_ip in selected_servers
        #     }
            
        for future in concurrent.futures.as_completed(futures, timeout=10):
            server_ip = futures[future]
            try:
                res = future.result()
                if res is not None:
                    logging.info(f"[{self.ip}]: Completed request to {server_ip}")
            except Exception as e:
                logging.error(f"[{self.ip}]: Request to {server_ip} generated an exception: {e}")
                logging.error(traceback.format_exc())

        self.flowtracker.complete_query(query_id, list(flow_ids.values()))

    def start(self):
        """Generate incast queries at the specified QPS."""
        interval = 1 / self.qps  # Interval between queries
        start_time = time.time()
        while time.time() - start_time < self.duration:
            self.send_query()
            time.sleep(interval)
        
        logging.info(f"[{self.ip}]: Completed all queries in {self.duration} seconds.")
        self.executor.shutdown(wait=False)


class BackgroundClient(BaseClient):
    """
    TODO scale the inter arrival times (sending rate) to match the background load
    TODO moderator should signal the end of the experiment to avoid pending flows (NULL values in the metrics db)
    """
    def __init__(self, server_ips, flow_ids, flow_sizes, inter_arrival_times, duration, congestion_control='cubic', exp_id=''):
        super().__init__(congestion_control, exp_id)
        self.duration = duration
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
                # logging.info(f"Connecting to {server_ip}:12345")
                s.connect((server_ip, 12345))
                # Prepare the data payload with the flow ID prefixed
                flow_id_prefix = f"{str(flow_id):<8}".encode('utf-8')[:8]
                data = flow_id_prefix + (b'x' * flow_size)
                s.sendall(data)  # Send all data in one go
                self.flowtracker.start_flow(str(flow_id), self.ip, server_ip, flow_size, flow_type='background')
                # logging.info(f"Sending flow {flow_id} to {server_ip}:12345")
            except Exception as e:
                logging.error(f"[{self.ip}]: Error sending traffic to {server_ip}: {e}")
                logging.debug(f"[{self.ip}]: Closing connection...")
                s.close()
                traceback.print_exc()
                sys.exit(1)

    def start(self):
        """Send flows until the specified duration (in seconds) elapses.
        Any flow in the middle of sending will be allowed to complete.
        """
        start_time = time.time()
        logging.debug(f"Client {self.ip} sending {len(self.flow_ids)} flows for {self.duration} seconds")
        n_flows = len(self.flow_ids)
        counter = 0
        flow_id_step = 0

        # Continue to iterate until duration has passed.
        while time.time() - start_time < self.duration:
            for flow_id, inter_arrival_time, flow_size, server_ip in zip(
                self.flow_ids, self.inter_arrival_times, self.flow_sizes, self.server_ips
            ):
                # Check before starting a new flow if we have reached the duration.
                if time.time() - start_time >= self.duration:
                    break

                # Optionally extend the flow_id if we have looped through all flows.
                if counter >= n_flows:
                    flow_id = f"{flow_id}-{flow_id_step}"
                counter += 1

                # Send the flow; note that send_request is a blocking call
                self.send_request(str(flow_id), server_ip, flow_size)

                # Sleep for the given inter-arrival time.
                time.sleep(float(inter_arrival_time))
            flow_id_step += 1

        logging.info(f"Client {self.ip} finished sending flows after {self.duration} seconds.")


class IperfClient(BaseClient):
    def __init__(self, server_ip, duration):
        super().__init__()
        self.server_ip = server_ip
        self.duration = duration

    def start(self):
        cmd = f'iperf3 -c {self.server_ip} -t {self.duration} &'
        os.system(cmd)
