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
import queue


class BaseClient(ABC):
    def __init__(self, congestion_control='cubic', exp_id='', server_ip=None):
        self.ip = self.get_host_ip()
        self.server_ip = server_ip
        self.congestion_control = congestion_control
        self.exp_id = exp_id
        if self.exp_id:
            self.flowtracker = FlowMetricsManager(self.exp_id)
        # Add tcpdump process tracking
        self.tcpdump_process = None

    def start_packet_capture(self, port, capture_file=None):
        """Start tcpdump packet capture to a pcap file."""
        if not capture_file and self.exp_id:
            # Generate a descriptive filename if not provided
            client_type = self.__class__.__name__.lower()
            capture_file = f"tmp/{self.exp_id}/{client_type}_{self.ip}_{port}.pcap"
        
        if capture_file:
            try:
                logging.info(f"[{self.ip}]: Starting packet capture to {capture_file}")
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(capture_file), exist_ok=True)
                
                # Start tcpdump on the port, capturing full packets
                cmd = [
                    "tcpdump", 
                    "-i", "any",                  # Capture on any interface
                    "-w", capture_file,           # Write to pcap file
                    f"port {port}",               # Filter for specific port
                    "-s", "0"                     # Capture entire packets
                ]
                self.tcpdump_process = subprocess.Popen(cmd)
                logging.info(f"[{self.ip}]: Packet capture started (PID: {self.tcpdump_process.pid})")
            except Exception as e:
                logging.error(f"[{self.ip}]: Failed to start packet capture: {e}")
        else:
            logging.warning(f"[{self.ip}]: No capture file specified, packet capture disabled")
    
    def stop_packet_capture(self):
        """Stop the tcpdump packet capture process."""
        if self.tcpdump_process:
            try:
                logging.info(f"[{self.ip}]: Stopping packet capture")
                self.tcpdump_process.terminate()
                self.tcpdump_process.wait(timeout=3)
                logging.info(f"[{self.ip}]: Packet capture stopped")
            except subprocess.TimeoutExpired:
                logging.warning(f"[{self.ip}]: Timeout stopping packet capture, forcing kill")
                self.tcpdump_process.kill()
            except Exception as e:
                logging.error(f"[{self.ip}]: Error stopping packet capture: {e}")
            self.tcpdump_process = None

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

    def _generate_flow_id(self, src_ip, dst_ip, src_port, dst_port):
        """Generate a unique flow ID based on source/destination IP and ports."""
        return "".join(f"{src_ip}{src_port}{dst_ip}{dst_port}".split('.'))

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
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0) # Disable Nagle's algorithm
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
        
    def _create_packet(self, flow_id, seq, packet_type='background'):
        """Create a packet with flow_id, sequence number and packet type header."""
        flow_id_bytes = flow_id.to_bytes(4, byteorder='big')
        seq_bytes = seq.to_bytes(4, byteorder='big')
        # Add packet type (0 for background, 1 for burst traffic)
        type_byte = b'\x00' if packet_type == 'background' else b'\x01'
        payload_size = self.packet_size - 9  # flow_id(4) + seq(4) + type(1)
        payload = b'X' * payload_size
        return flow_id_bytes + seq_bytes + type_byte + payload
    
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
                 log_file=None, duration=None):
        
        super().__init__(congestion_control, exp_id, server_ips[0])
        self.server_ips = server_ips  # Now a list of servers
        self.server_port = server_port
        self.num_packets = num_packets
        self.interval = interval
        #self.num_flows = num_flows
        self.packet_size = packet_size
        self.duration = duration if duration is not None else float('inf')
        
        # Bursty traffic parameters
        self.burst_interval = burst_interval  # Time between bursts (seconds)
        self.burst_servers = min(burst_servers, len(server_ips))  # Number of servers in each burst
        self.burst_reply_size = burst_reply_size  # Size of response from each server
        
        # Background flow tracking
        self.background_flow_running = False
        self.highest_seq = {}  # For tracking sequence numbers and detecting reordering
        
        # Response logging setup
        self.log_file = log_file
        
        # Initialize the CSV log file for responses
        if self.log_file:
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["timestamp", "src_ip", "dst_ip", "port", "flow_id", "seq", "packet_size", "packet_type", "reordering_flag"])
        else:
            logging.warning("No CSV file specified for logging responses. Responses will not be logged.")
        
        logging.info(f"[{self.ip}] Initialized Mixed Collection Client targeting {len(server_ips)} servers")
        logging.info(f"[{self.ip}] Response logging to: {self.log_file}")
    
    def _create_packet(self, flow_id, seq, packet_type='background'):
        """Create a packet with flow_id, sequence number and packet type header."""
        flow_id_bytes = flow_id.to_bytes(4, byteorder='big')
        seq_bytes = seq.to_bytes(4, byteorder='big')
        # Add packet type (0 for background, 1 for burst traffic)
        type_byte = b'\x00' if packet_type == 'background' else b'\x01'
        if packet_type == 'request':
            return flow_id_bytes + seq_bytes + type_byte
        else:
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
    
    def _send_burst_request(self, src_port, server_ip, burst_id, server_idx):
        """Send a burst request to a server and measure response time."""
        try:
            start_time = time.time()
            total_bytes = 0
            
            # Create a TCP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Bind to the specified source port
                try:
                    s.bind((self.ip, src_port))
                    logging.debug(f"Bound to source port {src_port} for burst {burst_id} to {server_ip}")
                except socket.error as e:
                    logging.error(f"Binding to source port {src_port} failed: {e}")
                    # Try with another source port
                    new_src_port = src_port + 1000 + random.randint(1, 1000)
                    logging.warning(f"Retrying with port {new_src_port}")
                    s.bind((self.ip, new_src_port))
                
                # Set TCP congestion control and disable Nagle's algorithm
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Set MSS (Maximum Segment Size) to influence packet size
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)  # Common datacenter MSS
                
                # Limit the send buffer size to prevent large bursts
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)  # Moderate buffer size
                
                # Connect to server (use higher port for burst traffic)
                s.connect((server_ip, 12346))
                
                # Send request - let TCP handle segmentation
                s.sendall(b'REQUEST')
                
                # Receive burst response in chunks
                while True:
                    data = s.recv(4096)  # Receive in chunks
                    if not data:
                        break
                    total_bytes += len(data)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logging.debug(f"Completed burst request {burst_id} to {server_ip}: {total_bytes} bytes in {response_time:.3f}s")
            
            return {
                'server_ip': server_ip,
                'response_time': response_time,
                'bytes_received': total_bytes
            }
            
        except Exception as e:
            logging.error(f"Error in burst request to {server_ip}: {e}")
            logging.error(traceback.format_exc())
            return {
                'server_ip': server_ip,
                'response_time': None,
                'bytes_received': 0
            }
    
    def _background_flow_worker(self):
        """Worker thread to continuously send background flows."""
        self.background_flow_running = True
        while self.background_flow_running:
            #for flow_num in range(self.num_flows):
            # Generate random flow ID and select random server
            flow_id = random.randint(0, 999999)
            target_server = random.choice(self.server_ips)
            self._send_background_flow(flow_id, target_server)
            time.sleep(self.interval)  # Wait between flows
            if time.time() - self.start_time >= self.duration:
                break
        self.background_flow_running = False
        logging.info(f"Background flow worker stopping after {self.duration} seconds")
        time.sleep(0.1)  # Allow any remaining flows to finish

    
    def _burst_worker(self):
        """Worker thread to periodically send burst requests."""
        while self.background_flow_running:  # Use the same flag to control both threads
            self._send_burst_request(0, random.choice(self.server_ips), random.randint(1000000, 9999999), 0)
            time.sleep(self.burst_interval)
    
    def start(self):
        """Start sending both background flows and periodic bursts."""
        logging.info(f"Starting mixed traffic with background flows and bursts every {self.burst_interval}s")
        
        # Start background flow thread
        bg_thread = threading.Thread(target=self._background_flow_worker)
        bg_thread.daemon = True
        bg_thread.start()
        
        time.sleep(0.1)
        
        # Start burst thread
        burst_thread = threading.Thread(target=self._burst_worker)
        burst_thread.daemon = True
        burst_thread.start()

        self.start_time = time.time() 
        
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

class BackgroundTcpClient(BaseClient):
    """Client for sending background TCP traffic with optimized buffered logging."""
    def __init__(self, server_ips, flow_size=1000000, flow_iat=0.1,
                 congestion_control='cubic', exp_id='', duration=None, 
                 capture_pcap=True, log_buffer_size=1000, log_flush_interval=1.0):
        
        super().__init__(congestion_control, exp_id)
        self.server_ips = server_ips  # List of server IPs
        self.flow_size = flow_size    # Total bytes to send per flow
        self.duration = duration if duration is not None else float('inf')
        self.log_file = f"tmp/{exp_id}/bg_client_{self.ip}_12345.csv" if exp_id else None
        self.background_flow_running = False
        self.source_port = 20000  # Starting source port for flows
        self.interval = flow_iat
        self.capture_pcap = capture_pcap
        
        # Logging parameters
        self.log_buffer_size = log_buffer_size
        self.log_flush_interval = log_flush_interval
        
        # Initialize logging system if a log file is provided
        if self.log_file:
            # Create a queue for log entries
            self.log_queue = queue.Queue()
            
            # Create directory for log file
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            # Initialize CSV file with headers
            with open(self.log_file, 'w', newline='', buffering=8192) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["flow_id", "start_time", "end_time", "flow_completion_time", 
                                "src_ip", "dst_ip", "src_port", "dst_port", "flow_size", "congestion_control"])
            
            # Start background logging thread
            self.log_thread_running = True
            self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.log_thread.start()
            
            logging.info(f"[{self.ip}] Initialized buffered logging with buffer size {self.log_buffer_size}")
        
        logging.info(f"[{self.ip}] Initialized Background TCP Client targeting {len(server_ips)} servers")
        logging.info(f"[{self.ip}] Using congestion control algorithm: {self.congestion_control}")
        logging.info(f"[{self.ip}] Flow size: {self.flow_size} bytes (letting TCP handle segmentation)")
    
    def _send_background_flow(self, target_server):
        """Send a single background TCP flow of specified size, letting TCP handle segmentation."""
        # Increment the source port for each flow to ensure uniqueness
        self.source_port += 1
        src_port = self.source_port
        dst_port = 12345  # Default destination port for background traffic
        
        # Generate the unique flow ID
        flow_id = self._generate_flow_id(self.ip, target_server, src_port, dst_port)
        
        try:
            start_time = time.time()
            # Create a TCP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Bind the socket to the specific source port
                s.bind((self.ip, src_port))
                
                # Set TCP congestion control and disable Nagle's algorithm
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Set MSS (Maximum Segment Size) to influence packet size
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)  # Common datacenter MSS
                
                # Limit the send buffer size to prevent large bursts
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)  # Moderate buffer size
                
                # Disable delayed ACKs
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                
                # Connect to server
                s.connect((target_server, dst_port))
                
                # Send data in one go - let TCP handle segmentation
                s.sendall(b'X' * self.flow_size)
            
            # We compute the FCT considering the TCP handshake as packets RTT will be computed using TCP timestamps
            end_time = time.time()
            flow_completion_time = end_time - start_time
            # Add flow details to log queue
            if hasattr(self, 'log_queue'):
                self.log_queue.put([
                    flow_id, 
                    start_time, 
                    end_time,
                    flow_completion_time,
                    self.ip, 
                    target_server, 
                    src_port, 
                    dst_port, 
                    self.flow_size,
                    self.congestion_control
                ])
            
            logging.debug(f"Flow {flow_id}: {self.flow_size} bytes, FCT: {flow_completion_time:.3f}s")
            
            return flow_completion_time
            
        except Exception as e:
            logging.error(f"Error in background flow {flow_id}: {e}")
            logging.error(traceback.format_exc())
            return None
    
    def _log_worker(self):
        """Background thread for asynchronous logging with buffering."""
        log_buffer = []
        last_flush_time = time.time()
        
        while self.log_thread_running:
            try:
                # Try to get an item from the queue with timeout to allow periodic flushing
                try:
                    log_entry = self.log_queue.get(timeout=0.1)
                    log_buffer.append(log_entry)
                    self.log_queue.task_done()
                except queue.Empty:
                    # No new entries, check if we need to flush based on time
                    pass
                
                # Check if we should flush based on buffer size or time interval
                current_time = time.time()
                time_since_flush = current_time - last_flush_time
                
                if (len(log_buffer) >= self.log_buffer_size or 
                    time_since_flush >= self.log_flush_interval) and log_buffer:
                    
                    # Flush buffer to disk
                    with open(self.log_file, 'a', newline='', buffering=8192) as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(log_buffer)
                    
                    # Clear buffer and reset timer
                    buffer_size = len(log_buffer)
                    log_buffer = []
                    last_flush_time = current_time
                    
                    logging.debug(f"Flushed {buffer_size} log entries to disk")
            
            except Exception as e:
                logging.error(f"Error in log worker: {e}")
                logging.error(traceback.format_exc())
        
        # Final flush when thread is shutting down
        if log_buffer:
            try:
                with open(self.log_file, 'a', newline='', buffering=8192) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(log_buffer)
                logging.debug(f"Final flush: Wrote {len(log_buffer)} log entries to disk")
            except Exception as e:
                logging.error(f"Error in final log flush: {e}")
    
    def _background_flow_worker(self):
        """Worker thread to continuously send background flows."""
        self.background_flow_running = True
        self.start_time = time.time()
        flow_count = 0
        
        while self.background_flow_running:
            # Select random server for this flow
            target_server = random.choice(self.server_ips)
            fct = self._send_background_flow(target_server)
            
            if fct is not None:
                flow_count += 1
            
            # Check if duration reached
            if time.time() - self.start_time >= self.duration:
                break

            # Inter-arrival time between flows
            time.sleep(self.interval)
        
        self.background_flow_running = False
        logging.info(f"Background flow worker completed {flow_count} flows in {time.time() - self.start_time:.2f}s")
    
    def start(self):
        """Start sending background TCP traffic."""
        logging.info(f"Starting background TCP traffic for {self.duration}s")
        
        # Start packet capture if enabled
        if self.capture_pcap and self.exp_id:
            pcap_file = f"tmp/{self.exp_id}/bg_client_{self.ip}_12345.pcap"
            self.start_packet_capture(12345, pcap_file)
        
        # Start background flow thread
        bg_thread = threading.Thread(target=self._background_flow_worker)
        bg_thread.daemon = True
        bg_thread.start()
        
        # Keep main thread alive until duration reached
        try:
            bg_thread.join()
        except KeyboardInterrupt:
            logging.info("Stopping client threads...")
            self.background_flow_running = False
        finally:
            # Signal log thread to terminate and flush remaining entries
            if hasattr(self, 'log_thread') and self.log_thread.is_alive():
                logging.info("Waiting for log flush...")
                self.log_thread_running = False
                # Wait a short time for final flush
                self.log_thread.join(timeout=2)
            
            self.stop_packet_capture()


class BurstyTcpClient(BaseClient):
    """Client for sending bursty TCP traffic with efficient QCT logging."""
    def __init__(self, server_ips, burst_interval=1.0, burst_servers=2, burst_reply_size=40000,
                 congestion_control='cubic', exp_id='', duration=None, 
                 capture_pcap=True, log_buffer_size=1000, log_flush_interval=1.0):
        super().__init__(congestion_control, exp_id)
        self.server_ips = server_ips  # List of server IPs
        self.burst_interval = burst_interval  # Time between bursts
        self.burst_servers = min(burst_servers, len(server_ips))  # Number of servers in each burst
        self.burst_reply_size = burst_reply_size  # Size of expected response
        self.duration = duration if duration is not None else float('inf')
        self.log_file = f"tmp/{exp_id}/bursty_client_{self.ip}_12345.csv" if exp_id else None
        self.burst_running = False
        self.capture_pcap = capture_pcap
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(server_ips))
        
        self.log_buffer_size = log_buffer_size  # Max entries before flush
        self.log_flush_interval = log_flush_interval  # Seconds between forced flushes
        
        if self.log_file:
            self.log_queue = queue.Queue()
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, 'w', newline='', buffering=8192) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "burst_id", "timestamp", "qct", "num_servers", "total_bytes_received",
                    "src_ip", "congestion_control", "slowest_server", "slowest_server_time"
                ])
            self.log_thread_running = True
            self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
            self.log_thread.start()
            logging.info(f"[{self.ip}] Initialized buffered logging with buffer size {self.log_buffer_size}")
        
        logging.info(f"[{self.ip}] Initialized Bursty TCP Client targeting {len(server_ips)} servers")
        logging.info(f"[{self.ip}] Using congestion control algorithm: {self.congestion_control}")
        
        # Initialize the source port counter
        self.base_src_port = 50000
        self.source_port = self.base_src_port  # Initialize to base value

    def _send_burst_request(self, src_port, server_ip, burst_id, server_idx):
        """Send a burst request to a server and measure response time."""
        try:
            start_time = time.time()
            total_bytes = 0
            
            # Create a TCP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                # Bind to the specified source port
                try:
                    s.bind((self.ip, src_port))
                    logging.debug(f"Bound to source port {src_port} for burst {burst_id} to {server_ip}")
                except socket.error as e:
                    logging.error(f"Binding to source port {src_port} failed: {e}")
                    # Try with another source port
                    new_src_port = src_port + 1000 + random.randint(1, 1000)
                    logging.warning(f"Retrying with port {new_src_port}")
                    s.bind((self.ip, new_src_port))
                
                # Set TCP congestion control and disable Nagle's algorithm
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Set MSS (Maximum Segment Size) to influence packet size
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)  # Common datacenter MSS
                
                # Limit the send buffer size to prevent large bursts
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)  # Moderate buffer size
                
                # Connect to server (use higher port for burst traffic)
                s.connect((server_ip, 12346))
                
                # Send request - let TCP handle segmentation
                s.sendall(b'REQUEST')
                
                # Receive burst response in chunks
                while True:
                    data = s.recv(4096)  # Receive in chunks
                    if not data:
                        break
                    total_bytes += len(data)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            logging.debug(f"Completed burst request {burst_id} to {server_ip}: {total_bytes} bytes in {response_time:.3f}s")
            
            return {
                'server_ip': server_ip,
                'response_time': response_time,
                'bytes_received': total_bytes
            }
            
        except Exception as e:
            logging.error(f"Error in burst request to {server_ip}: {e}")
            logging.error(traceback.format_exc())
            return {
                'server_ip': server_ip,
                'response_time': None,
                'bytes_received': 0
            }
    
    def _send_burst(self, base_src_port, burst_id):
        """Send burst requests to multiple servers concurrently and track QCT."""
        
        # Select random servers for this burst
        target_servers = random.sample(self.server_ips, self.burst_servers)
        logging.info(f"Sending burst {burst_id} to {len(target_servers)} servers starting from source port {base_src_port}")
        
        # Record burst start time
        burst_start_time = time.time()
        
        # Send requests concurrently
        futures = []
        for idx, server_ip in enumerate(target_servers):
            # Use different source port for each server in this burst
            src_port = base_src_port + idx
            futures.append(self.executor.submit(
                self._send_burst_request, 
                src_port,  # Different source port for each server
                server_ip, 
                burst_id, 
                idx
            ))
        
        # Wait for all responses and collect results
        responses = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result['response_time'] is not None:
                    responses.append(result)
            except Exception as e:
                logging.error(f"Exception in burst request: {e}")
        
        # Calculate overall QCT (Query Completion Time)
        burst_end_time = time.time()
        qct = burst_end_time - burst_start_time
        
        # Find the slowest server in this burst
        slowest_server = None
        slowest_time = 0
        total_bytes = 0
        
        for resp in responses:
            total_bytes += resp['bytes_received']
            if resp['response_time'] > slowest_time:
                slowest_time = resp['response_time']
                slowest_server = resp['server_ip']
        
        # Log burst QCT details to queue
        if hasattr(self, 'log_queue'):
            self.log_queue.put([
                burst_id,                    # Unique burst ID
                burst_start_time,            # When burst started
                qct,                         # Total burst completion time
                len(target_servers),         # Number of servers in burst
                total_bytes,                 # Total bytes received
                self.ip,                     # Client IP
                self.congestion_control,     # Congestion control algorithm
                slowest_server,              # IP of slowest responding server 
                slowest_time                 # Time taken by slowest server
            ])
        
        logging.debug(f"Completed burst {burst_id}: QCT={qct:.3f}s, {len(responses)}/{len(target_servers)} servers responded")
        return qct
    
    def _burst_worker(self):
        """Worker thread to periodically send bursts."""
        self.burst_running = True
        self.start_time = time.time()
        burst_count = 0
        
        while self.burst_running:
            # Generate burst_id
            burst_id = f"burst_{self.ip}_{burst_count}"
            
            # Calculate base source port for this burst
            # Increment by burst_servers to ensure enough ports for next burst
            base_src_port = self.base_src_port + (burst_count * self.burst_servers)
            
            # Send burst using the current base source port
            qct = self._send_burst(base_src_port, burst_id)
            burst_count += 1
            
            # Sleep to maintain consistent burst intervals
            elapsed = time.time() - self.start_time
            next_burst_time = self.start_time + (burst_count * self.burst_interval)
            sleep_time = max(0, next_burst_time - time.time())
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Check if we've reached the duration limit
            if time.time() - self.start_time >= self.duration:
                break
        
        self.burst_running = False
        elapsed_time = time.time() - self.start_time
        logging.info(f"Burst worker completed {burst_count} bursts in {elapsed_time:.2f}s")
        logging.info(f"Average QCT: {elapsed_time/max(1, burst_count):.3f}s")
    
    def _log_worker(self):
        """Background thread for asynchronous logging with buffering."""
        log_buffer = []
        last_flush_time = time.time()
        
        while self.log_thread_running:
            try:
                # Try to get an item from the queue with timeout to allow periodic flushing
                try:
                    log_entry = self.log_queue.get(timeout=0.1)
                    log_buffer.append(log_entry)
                    self.log_queue.task_done()
                except queue.Empty:
                    # No new entries, check if we need to flush based on time
                    pass
                
                # Check if we should flush based on buffer size or time interval
                current_time = time.time()
                time_since_flush = current_time - last_flush_time
                
                if (len(log_buffer) >= self.log_buffer_size or 
                    time_since_flush >= self.log_flush_interval) and log_buffer:
                    
                    # Flush buffer to disk
                    with open(self.log_file, 'a', newline='', buffering=8192) as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(log_buffer)
                    
                    # Clear buffer and reset timer
                    buffer_size = len(log_buffer)
                    log_buffer = []
                    last_flush_time = current_time
                    
                    logging.debug(f"Flushed {buffer_size} log entries to disk")
            
            except Exception as e:
                logging.error(f"Error in log worker: {e}")
                logging.error(traceback.format_exc())
        
        # Final flush when thread is shutting down
        if log_buffer:
            try:
                with open(self.log_file, 'a', newline='', buffering=8192) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(log_buffer)
                logging.debug(f"Final flush: Wrote {len(log_buffer)} log entries to disk")
            except Exception as e:
                logging.error(f"Error in final log flush: {e}")
    
    def start(self):
        """Start sending bursty TCP traffic."""
        logging.info(f"Starting bursty TCP traffic with {self.burst_interval}s interval for {self.duration}s")
        
        # Start packet capture if enabled
        if self.capture_pcap and self.exp_id:
            pcap_file = f"tmp/{self.exp_id}/bursty_client_{self.ip}_12346.pcap"
            self.start_packet_capture(12346, pcap_file)
        
        # Start burst thread
        burst_thread = threading.Thread(target=self._burst_worker)
        burst_thread.daemon = True
        burst_thread.start()
        
        # Keep main thread alive until duration reached
        try:
            burst_thread.join()
        except KeyboardInterrupt:
            logging.info("Stopping client threads...")
            self.burst_running = False
            self.executor.shutdown(wait=False)
        finally:
            # Signal log thread to terminate and flush remaining entries
            if hasattr(self, 'log_thread') and self.log_thread.is_alive():
                logging.info("Waiting for log flush...")
                self.log_thread_running = False
                # Wait a short time for final flush
                self.log_thread.join(timeout=2)
            
            self.stop_packet_capture()
