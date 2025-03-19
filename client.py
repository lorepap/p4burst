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
import csv
import struct

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

class TestCollectionClient(BaseClient):
    """
    Client for sending UDP packets with custom headers for flow tracking.
    This client uses sockets to send packets with flow_id and sequence number headers.
    """
    def __init__(self, server_ip, server_port=12345, num_packets=1000, interval=0.001, 
                 num_flows=1, congestion_control='cubic', exp_id='', packet_size=1024):
        
        super().__init__(congestion_control, exp_id, server_ip)
        self.server_port = server_port
        self.num_packets = num_packets
        self.interval = interval
        self.num_flows = num_flows
        self.packet_size = packet_size
        logging.info(f"Initialized Collection Client targeting {server_ip}:{server_port}")
        
    def _create_packet(self, flow_id, seq):
        """Create a packet with flow_id and sequence number header."""
        # Create a binary packet with our custom header:
        # [flow_id (4 bytes)][seq (4 bytes)][payload]
        flow_id_bytes = flow_id.to_bytes(4, byteorder='big')
        seq_bytes = seq.to_bytes(4, byteorder='big')
        # payload = b'X' * 64 
        payload = b'X' * (self.packet_size - 8)
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

class DataCollectionClient(BaseClient):
    """
    Client for sending both background UDP flows and bursty incast traffic.
    """
    def __init__(self, server_ips, server_port=12345, num_packets=1000, interval=0.001,
                 congestion_control='cubic', exp_id='', packet_size=1000,
                 burst_interval=1.0, burst_servers=2, burst_reply_size=40000,
                 client_id=0, log_dir=''):
        
        super().__init__(congestion_control, exp_id, server_ips[0])
        self.server_ips = server_ips  # Now a list of servers
        self.server_port = server_port
        self.num_packets = num_packets
        self.interval = interval
        #self.num_flows = num_flows
        self.packet_size = packet_size
        
        # Bursty traffic parameters
        self.burst_interval = burst_interval  # Time between bursts (seconds)
        self.burst_servers = min(burst_servers, len(server_ips))  # Number of servers in each burst
        self.burst_reply_size = burst_reply_size  # Size of response from each server
        
        # Background flow tracking
        self.background_flow_running = False
        self.highest_seq = {}  # For tracking sequence numbers and detecting reordering
        
        # Response logging setup
        self.client_id = client_id
        self.log_dir = log_dir if log_dir else os.path.dirname(exp_id)
        self.log_file = f"{self.log_dir}/sender_{self.client_id}_dataset.csv"
        
        # Initialize the CSV log file for responses
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "src_ip", "dst_ip", "port", "flow_id", "seq", "packet_type", "reordering_flag"])
        
        logging.info(f"[{self.ip}] Initialized Mixed Collection Client targeting {len(server_ips)} servers")
        logging.info(f"[{self.ip}] Response logging to: {self.log_file}")
    
    def _create_packet(self, flow_id, seq, packet_type='background'):
        """Create a packet with flow_id, sequence number and packet type header."""
        flow_id_bytes = flow_id.to_bytes(4, byteorder='big')
        seq_bytes = seq.to_bytes(4, byteorder='big')
        # Add packet type (0 for background, 1 for request, 2 for response)
        type_byte = b'\x00' if packet_type == 'background' else (b'\x01' if packet_type == 'request' else b'\x02')
        payload_size = self.packet_size - 9  # flow_id(4) + seq(4) + type(1)
        payload = b'X' * payload_size
        return flow_id_bytes + seq_bytes + type_byte + payload
    
    def _send_background_flow(self, flow_id, target_server):
        """Send a single background flow with sequential packets."""
        logging.info(f"Sending background flow {flow_id} with {self.num_packets} packets to {target_server}")
        
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            for seq in range(self.num_packets):
                try:
                    packet = self._create_packet(flow_id, seq, 'background')
                    s.sendto(packet, (target_server, self.server_port))
                    time.sleep(self.interval)
                except Exception as e:
                    logging.error(f"Error sending background packet of flow {flow_id}: {e}")
                    return
        logging.info(f"Completed background flow {flow_id}")
    
    def _send_burst_request(self):
        """Send burst requests to multiple servers and receive responses."""
        burst_id = random.randint(1000000, 9999999)
        # Select random subset of servers for this burst
        target_servers = random.sample(self.server_ips, self.burst_servers)
        logging.info(f"Sending burst request {burst_id} to {len(target_servers)} servers")
        
        # Send request packets to each server and wait for responses
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(target_servers)) as executor:
            futures = []
            for idx, server_ip in enumerate(target_servers):
                futures.append(executor.submit(
                    self._send_burst_request_and_wait, 
                    burst_id, 
                    idx,
                    server_ip
                ))
        
        logging.info(f"Completed burst request {burst_id}")
    
    def _send_burst_request_and_wait(self, burst_id, seq, server_ip):
        """Send a burst request packet to a server and wait for response."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Set timeout for receiving response
                s.settimeout(3)
                
                # Send request packet
                packet = self._create_packet(burst_id, seq, 'request')
                s.sendto(packet, (server_ip, self.server_port))
                
                # Wait for response
                try:
                    response, addr = s.recvfrom(self.burst_reply_size)
                    self._process_burst_response(response, addr)
                except socket.timeout:
                    logging.warning(f"Timeout waiting for response from {server_ip} for burst {burst_id}")
                    
        except Exception as e:
            logging.error(f"Error in burst request/response with {server_ip}: {e}")
            logging.error(traceback.format_exc())
    
    def _process_burst_response(self, data, addr):
        """Process and log a received burst response."""
        try:
            # Check packet length
            if len(data) < 9:  # flow_id(4) + seq(4) + type(1)
                logging.warning(f"Received malformed response (too short): {len(data)} bytes")
                return
            
            # Extract flow_id, seq, and verify packet type
            flow_id, seq = struct.unpack("!II", data[0:8])
            packet_type = data[8]
            
            if packet_type != 2:  # Should be response type
                logging.warning(f"Received non-response packet type: {packet_type}")
                return
                
            arrival_time = time.time()
            src_ip = addr[0]
            dst_ip = self.ip
            dport = self.server_port
            
            # Define a flow key (using src, dst, port, flow_id)
            key = (src_ip, dst_ip, dport, flow_id)
            
            # Check for packet reordering
            reorder_flag = 0
            if key in self.highest_seq:
                if seq < self.highest_seq[key]:
                    reorder_flag = 1  # Out-of-order packet
                else:
                    self.highest_seq[key] = seq
            else:
                self.highest_seq[key] = seq
            
            # Log response details to CSV
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([arrival_time, src_ip, dst_ip, dport, flow_id, seq, packet_type, reorder_flag])
                
            logging.debug(f"Logged burst response from {src_ip} for flow {flow_id}, seq {seq}")
            
        except Exception as e:
            logging.error(f"Error processing burst response from {addr[0]}:{addr[1]}: {e}")
            logging.error(traceback.format_exc())
    
    def _background_flow_worker(self):
        """Worker thread to continuously send background flows."""
        self.background_flow_running = True
        while self.background_flow_running:
            #for flow_num in range(self.num_flows):
            # Generate random flow ID and select random server
            flow_id = random.randint(0, 2**32-1)
            target_server = random.choice(self.server_ips)
            self._send_background_flow(flow_id, target_server)
            time.sleep(self.interval * self.num_packets)  # Wait between flows
    
    def _burst_worker(self):
        """Worker thread to periodically send burst requests."""
        while self.background_flow_running:  # Use the same flag to control both threads
            self._send_burst_request()
            time.sleep(self.burst_interval)
    
    def start(self):
        """Start sending both background flows and periodic bursts."""
        logging.info(f"Starting mixed traffic with background flows and bursts every {self.burst_interval}s")
        
        # Start background flow thread
        bg_thread = threading.Thread(target=self._background_flow_worker)
        bg_thread.daemon = True
        bg_thread.start()
        
        # Start burst thread
        burst_thread = threading.Thread(target=self._burst_worker)
        burst_thread.daemon = True
        burst_thread.start()
        
        # Keep running until interrupted
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Stopping client threads...")
            self.background_flow_running = False
            # Give threads time to finish
            time.sleep(1)