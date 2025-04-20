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
                s.connect((server_ip, self.port))
                logging.info(f"[{self.ip}]: Sending single packet to {server_ip}:{self.port}")

                # Prepare the data packet with a specified size
                data = b'P' * packet_size  # Placeholder byte array for the packet
                logging.info("Sending packet...")
                s.sendall(data)  # Send all data in one go

                # Read the response from the server
                response = s.recv(1024)
                logging.info(f"Received response: {response}")

                logging.info(f"[{self.ip}]: Sent {len(data)} bytes to {server_ip}:{self.port}")
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


class BackgroundTcpClient(BaseClient):
    """Client for sending background TCP traffic with optimized buffered logging."""
    def __init__(self, server_ips, flow_size=1000000, flow_iat=0.1,
                 congestion_control='cubic', exp_id='', duration=None, 
                 capture_pcap=True, log_buffer_size=1000, log_flush_interval=1.0, port=12345):
        
        super().__init__(congestion_control, exp_id)
        self.server_ips = server_ips  # List of server IPs
        self.flow_size = flow_size    # Total bytes to send per flow
        self.duration = duration if duration is not None else float('inf')
        self.port = port
        self.log_file = f"tmp/{exp_id}/bg_client_{self.ip}_{self.port}.csv" if exp_id else None
        self.background_flow_running = False
        self.capture_pcap = capture_pcap
        self.source_port = 20000  # Starting source port for flows
        self.interval = flow_iat
        '''
        self.total_flows_sent = 0
        self.successful_flows = 0
        self.failed_flows = 0
        self.stats_lock = threading.Lock()
        '''
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
        dst_port = self.port  # Default destination port for background traffic
        
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
                    '''
                    with open(f"tmp/{self.exp_id}/bg_client_{self.ip}_completed_flows.csv", 'w', newline='', buffering=8192) as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(["total_flows_sent","successful_flows","failed_flows"])
                        writer.writerow([self.total_flows_sent, self.successful_flows, self.failed_flows])
                    '''
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
            #self.total_flows_sent += 1
            fct = self._send_background_flow(target_server)
            '''
            if fct is not None:
                self.successful_flows += 1
            else:
                self.failed_flows += 1
            
            if fct is not None:
                flow_count += 1
            '''
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
            pcap_file = f"tmp/{self.exp_id}/bg_client_{self.ip}_{self.port}.pcap"
            self.start_packet_capture(self.port, pcap_file)
        
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
                 capture_pcap=True, log_buffer_size=1000, log_flush_interval=1.0, port=12346):
        super().__init__(congestion_control, exp_id)
        self.server_ips = server_ips  # List of server IPs
        self.burst_interval = burst_interval  # Time between bursts
        self.burst_servers = min(burst_servers, len(server_ips))  # Number of servers in each burst
        self.burst_reply_size = burst_reply_size  # Size of expected response
        self.duration = duration if duration is not None else float('inf')
        self.port = port
        self.log_file = f"tmp/{exp_id}/bursty_client_{self.ip}_{self.port}.csv" if exp_id else None
        self.burst_running = False
        self.capture_pcap = capture_pcap
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(server_ips))

        logging.info(f"burst reply size: {self.burst_reply_size}, burst_intervall: {self.burst_interval}, burst_servers: {self.burst_servers}")
        
        self.log_buffer_size = log_buffer_size  # Max entries before flush
        self.log_flush_interval = log_flush_interval  # Seconds between forced flushes
        '''
        self.total_queries_sent = 0
        self.successful_queries = 0
        self.failed_queries = 0
        '''
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
                s.connect((server_ip, self.port))
                
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
        
        #self.total_queries_sent += 1
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
        '''
        if qct:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        '''
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
                '''
                with open(f"tmp/{self.exp_id}/bursty_client_{self.ip}_completed_queries.csv", 'w', newline='', buffering=8192) as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["total_queries_sent","successful_queries","failed_queries"])
                    writer.writerow([self.total_queries_sent, self.successful_queries, self.failed_queries])
                '''
                logging.debug(f"Final flush: Wrote {len(log_buffer)} log entries to disk")
            except Exception as e:
                logging.error(f"Error in final log flush: {e}")
    
    def start(self):
        """Start sending bursty TCP traffic."""
        logging.info(f"Starting bursty TCP traffic with {self.burst_interval}s interval for {self.duration}s")
        
        # Start packet capture if enabled
        if self.capture_pcap and self.exp_id:
            pcap_file = f"tmp/{self.exp_id}/bursty_client_{self.ip}_{self.port}.pcap"
            self.start_packet_capture(self.port, pcap_file)
        
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
