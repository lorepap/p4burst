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

class CollectionClient(BaseClient):
    """
    Client for sending UDP packets with custom headers for flow tracking.
    This client uses sockets to send packets with flow_id and sequence number headers.
    """
    def __init__(self, server_ip, server_port=12345, num_packets=1000, interval=0.001, 
                 num_flows=1, congestion_control='cubic', exp_id=''):
        super().__init__(congestion_control, exp_id, server_ip)
        self.server_port = server_port
        self.num_packets = num_packets
        self.interval = interval
        self.num_flows = num_flows
        logging.info(f"Initialized Collection Client targeting {server_ip}:{server_port}")
        
    def _create_packet(self, flow_id, seq):
        """Create a packet with flow_id and sequence number header."""
        # Create a binary packet with our custom header:
        # [flow_id (4 bytes)][seq (4 bytes)][payload]
        flow_id_bytes = flow_id.to_bytes(4, byteorder='big')
        seq_bytes = seq.to_bytes(4, byteorder='big')
        payload = b'X' * 64 
        return flow_id_bytes + seq_bytes + payload
    
    def _send_flow(self, flow_id):
        """Send a single flow with sequential packets."""
        logging.info(f"Sending flow {flow_id} with {self.num_packets} packets")
        
        # Create a UDP socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            #s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
            for seq in range(self.num_packets):
                try:
                    #s.connect((self.server_ip, self.server_port))
                    packet = self._create_packet(flow_id, seq)
                    s.sendto(packet, (self.server_ip, self.server_port))
                    time.sleep(self.interval)
                except Exception as e:
                    #logging.error(f"Could not connect to {self.server_ip}:{self.server_port}: {e}")
                    logging.error(f"Error sending packet of flow {flow_id} to {self.server_ip}: {e}")
                    return
            logging.info(f"Completed flow {flow_id}")
    
    def start(self):
        """Start sending flows with specified parameters."""
        logging.info(f"Starting to send {self.num_flows} flows with {self.num_packets} packets each")
        logging.info(f"Packet interval: {self.interval}s")
        
        start_time = time.time()
        
        for flow_num in range(self.num_flows):
            # Generate random flow ID
            flow_id = random.randint(0, 2**32-1)
            logging.info(f"Starting flow {flow_num+1}/{self.num_flows} with ID {flow_id}")
            
            # If you have a flow tracker or metrics, register the flow here.
            self._send_flow(flow_id)
        
        total_runtime = time.time() - start_time
        logging.info(f"Finished sending all {self.num_flows} flows in {total_runtime:.2f} seconds")