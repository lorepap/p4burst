import socket
import time
from abc import ABC, abstractmethod
import logging
import traceback
import math
import csv
from metrics import FlowMetricsManager
import threading
import os
from scapy.all import sniff, Ether, IP, UDP, Packet, BitField, bind_layers
import threading
import struct
import subprocess
import signal
import sys

class BaseServer(ABC):
    def __init__(self, port=12345, ip=None, congestion_control='cubic', exp_id=''):
        self.port = int(port)
        self.congestion_control = congestion_control
        self.exp_id = exp_id
        # self.ip = self.get_host_ip()
        if ip:
            self.ip = ip 
        else: raise ValueError("IP address must be specified")

        if self.exp_id:
            self.flowtracker = FlowMetricsManager(self.exp_id)
        # Add tcpdump process tracking
        self.tcpdump_process = None
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logging.info(f"[{self.ip}]: Received shutdown signal {signum}")
        self.running = False
        self.stop_packet_capture()
        sys.exit(0)

    def handle_request(self, conn):
        try:
            # Receive data in chunks
            data = conn.recv(1024)
            logging.info(f"[{self.ip}]: Received {len(data)} bytes of data")
            
            # Echo back an acknowledgment or the received data
            if data:
                conn.sendall(b"ACK")  # Acknowledge the receipt of data

        except Exception as e:
            logging.error(f"[{self.ip}]: Error handling request: {e}")
        finally:
            logging.info(f"[{self.ip}]: Closing connection")
            conn.close()

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            s.listen()
            logging.info(f"[{self.ip}]: Server listening on port {self.port}")
            try:
                while self.running:
                    try:
                        conn, addr = s.accept()
                    except KeyboardInterrupt:
                        # Break out of the inner blocking call
                        raise
                    with conn:
                        print(f"Connected by {addr}")
                        self.handle_request(conn)
            except KeyboardInterrupt:
                logging.info(f"[{self.ip}]: Server shutting down gracefully.")
                self.stop_packet_capture()
            except Exception as e:
                logging.error(f"[{self.ip}]: Error accepting connection: {e}")
                logging.error(traceback.format_exc())
                self.stop_packet_capture()
            finally:
                self.stop_packet_capture()

    def start_packet_capture(self, capture_file=None):
        """Start tcpdump packet capture to a pcap file."""
        if not capture_file and self.exp_id:
            # Generate a descriptive filename if not provided
            server_type = self.__class__.__name__.lower()
            capture_file = f"tmp/{self.exp_id}/{server_type}_{self.ip}_{self.port}.pcap"
        
        if capture_file:
            try:
                logging.info(f"[{self.ip}]: Starting packet capture to {capture_file}")
                # Create directory if it doesn't exist and ensure it's writable
                capture_dir = os.path.dirname(capture_file)
                os.makedirs(capture_dir, exist_ok=True)
                # Ensure the directory is writable
                os.chmod(capture_dir, 0o777)
                
                # Start tcpdump on the port, capturing full packets
                cmd = [
                    "sudo", 
                    "-n",  # Non-interactive mode
                    "tcpdump", 
                    "-i", "any",                  # Capture on any interface
                    "-w", capture_file,           # Write to pcap file
                    f"port {self.port}",          # Filter for specific port
                    "-s", "0",                    # Capture entire packets
                    "-U",                         # Use buffered output
                    "-W", "1",                    # Use 1 buffer
                    "-C", "1000"                  # Rotate files at 1000MB
                ]
                
                # Run tcpdump with sudo and redirect stderr to capture any errors
                self.tcpdump_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setpgrp  # Create new process group
                )
                
                # Check if process started successfully
                if self.tcpdump_process.poll() is not None:
                    # Process terminated immediately, get error message
                    _, stderr = self.tcpdump_process.communicate()
                    error_msg = stderr.decode()
                    raise Exception(f"tcpdump failed to start: {error_msg}")
                
                logging.info(f"[{self.ip}]: Packet capture started (PID: {self.tcpdump_process.pid})")
            except Exception as e:
                logging.error(f"[{self.ip}]: Failed to start packet capture: {e}")
                # Try to clean up if process was created but failed
                if hasattr(self, 'tcpdump_process') and self.tcpdump_process:
                    try:
                        self.tcpdump_process.terminate()
                    except:
                        pass
                    self.tcpdump_process = None
        else:
            logging.warning(f"[{self.ip}]: No capture file specified, packet capture disabled")
    
    def stop_packet_capture(self):
        """Stop the tcpdump packet capture process."""
        if self.tcpdump_process:
            try:
                logging.info(f"[{self.ip}]: Stopping packet capture")
                # First try SIGTERM
                self.tcpdump_process.terminate()
                try:
                    # Wait for tcpdump to finish writing
                    self.tcpdump_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # If SIGTERM didn't work, try SIGINT
                    logging.warning(f"[{self.ip}]: SIGTERM timeout, trying SIGINT")
                    self.tcpdump_process.send_signal(signal.SIGINT)
                    try:
                        self.tcpdump_process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # If still not responding, force kill
                        logging.warning(f"[{self.ip}]: SIGINT timeout, forcing kill")
                        self.tcpdump_process.kill()
                        self.tcpdump_process.wait(timeout=1)
                
                logging.info(f"[{self.ip}]: Packet capture stopped")
            except Exception as e:
                logging.error(f"[{self.ip}]: Error stopping packet capture: {e}")
            finally:
                self.tcpdump_process = None

    def cleanup(self):
        """Clean up resources before exit."""
        try:
            self.stop_packet_capture()
            if hasattr(self, 'flowtracker'):
                self.flowtracker.close()
        except Exception as e:
            logging.error(f"[{self.ip}]: Error during cleanup: {e}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()


class IperfServer(BaseServer):
    def __init__(self, port=5201):
        self.port = port

    def start(self):
        cmd = f'iperf3 -s &'
        os.system(cmd)


class BurstyServer(BaseServer):
    def __init__(self, ip, reply_size, port=12346, exp_id='', cong_control='cubic'):
        super().__init__(port, ip, exp_id=exp_id, congestion_control=cong_control)
        self.reply_size = reply_size

    def start(self):
        """Start the server with multi-threading support."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen()
            logging.info(f"[{self.ip}]: Server listening on port {self.port}")

            while True:
                conn, addr = s.accept()
                # logging.info(f"[{self.ip}]: Accepted connection from {addr}")

                # Start a new thread for each client connection
                client_thread = threading.Thread(target=self.handle_request, args=(conn, addr))
                client_thread.daemon = True  # Daemon thread will exit when main program exits
                client_thread.start()

    def handle_request(self, conn, addr):
        """Handle a client request and send the response."""
        try:
            data = conn.recv(1024)
            if data:
                # Decode flow ID (optional, for metrics tracking)
                flow_id = data[:8].decode('utf-8') if len(data) >= 8 else None
                # TODO - This is not entirely accurate, as the QCT is computed at client side taking the minimum fct
                # In other words, the flow start time should be started at the client side
                # The problem is that the servers should send flows at the same time to properly simulate an incast event
                # TODO - Understand if the incast event is happening without the servers synchronization
                self.flowtracker.start_flow(flow_id, self.ip, addr[0], self.reply_size, flow_type='bursty')
                # Send the full reply size as one data stream
                conn.sendall(b'x' * self.reply_size)
        except Exception as e:
            logging.error(f"[{self.ip}]: Error handling request from {addr[0]}:{addr[1]}: {e}")
            logging.error(traceback.format_exc())
        finally:
            conn.close()


class BackgroundServer(BaseServer):
    def __init__(self, ip=None, port=12345, exp_id='', cong_control='cubic'):
        super().__init__(ip, port, cong_control, exp_id)

    def start(self):
        """Start the server with multi-threading support."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
            s.bind(('0.0.0.0', self.port))
            s.listen()
            # logging.info(f"[{self.ip}]: Server listening on port {self.port}")

            try:
                while True:
                    try:
                        conn, addr = s.accept()
                    except KeyboardInterrupt:
                        # Break out of the inner blocking call
                        raise
                    # logging.info(f"[{self.ip}]: Accepted connection from {addr}")

                    # with conn:
                    #     self.handle_request(conn, addr)
                    
                    client_thread = threading.Thread(target=self.handle_request, args=(conn, addr))
                    client_thread.daemon = True  # Daemon thread will exit when main program exits
                    client_thread.start()
            except KeyboardInterrupt:
                logging.info(f"[{self.ip}]: Server shutting down gracefully.")
            except Exception as e:
                logging.error(f"[{self.ip}]: Error accepting connection: {e}")
                logging.error(traceback.format_exc())


    def handle_request(self, conn, addr):
        """Handle incoming data, compute metrics, and send acknowledgment."""
        total_bytes_received = 0
        flow_id = None  # Assuming each connection has a unique flow_id assigned by the client
        try:
            data = conn.recv(4096)
            if data:
                flow_id = data[:8].decode('utf-8').strip().split()[0]
                total_bytes_received += len(data) - 8
                # print(f"Received data from {addr[0]}:{addr[1]} | Flow ID: {flow_id} | Bytes: {len(data) - 8}")
                # logging.info(f"Started Flow ID: {flow_id} | Bytes received: {total_bytes_received}")
            while data:
                data = conn.recv(4096)
                if not data:
                    break
                total_bytes_received += len(data)

            # Mark the flow as complete in FlowMetricsManager
            if flow_id is not None:
                self.flowtracker.complete_flow(flow_id)
                # logging.info(f"Flow {flow_id} from {addr} completed. Total bytes received: {total_bytes_received}")

        except Exception as e:
            logging.error(f"[{self.ip}]: Error receiving data from {addr[0]}:{addr[1]}: {e}")

    def get_metrics(self):
        """Retrieve collected flow metrics."""
        return self.flowtracker.get_metrics()

class TestCollectionServer(BaseServer):
    def __init__(self, ip=None, port=12345, exp_id='', log_file='receiver_log.csv'):
        super().__init__(port, ip, exp_id=exp_id)
        self.log_file = log_file
        self.highest_seq = {}  # Store the highest sequence number seen per flow.

        # Initialize the CSV log file
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "src_ip", "dst_ip", "port", "flow_id", "seq", "reordering_flag"])
            
        logging.info(f"[{self.ip}]: Collection server initialized with log file: {self.log_file}")

    def read_exactly(self, conn, nbytes):
        """
        Helper function to read exactly nbytes from the TCP connection.
        Returns the bytes read or None if the connection is closed prematurely.
        """
        data = b""
        while len(data) < nbytes:
            packet = conn.recv(nbytes - len(data))
            if not packet:
                return None
            data += packet
        return data

    def handle_connection(self, conn, addr):
        """Handle a single TCP connection."""
        src_ip = addr[0]
        dst_ip = self.ip
        port = self.port
        PACKET_SIZE = 72  # 4 bytes flow_id, 4 bytes seq, 64 bytes payload
        try:
            while True:
                data = self.read_exactly(conn, PACKET_SIZE)
                if data is None:
                    logging.info(f"[{self.ip}]: Connection closed by {src_ip}")
                    break
                self.process_packet(data, addr)
        except Exception as e:
            logging.error(f"[{self.ip}]: Error handling connection from {src_ip}: {e}")
            logging.error(traceback.format_exc())
        finally:
            conn.close()

    def start(self):
        """Start the TCP server for packet collection."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Set TCP socket options if needed
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            #s.listen()
            logging.info(f"[{self.ip}]: Collection server listening on UDP port {self.port}")

            try:
                while True:
                    try:
                        #conn, addr = s.accept()
                        data, addr = s.recvfrom(4096)
                        #logging.info(f"[{self.ip}]: Accepted connection from {addr[0]}:{addr[1]}")
                        # Handle each connection in a separate thread
                        #threading.Thread(target=self.handle_connection, args=(conn, addr), daemon=True).start()
                        self.process_packet(data, addr)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logging.error(f"[{self.ip}]: Error accepting connection: {e}")
                        logging.error(traceback.format_exc())
            except KeyboardInterrupt:
                logging.info(f"[{self.ip}]: Collection server shutting down gracefully.")

    def process_packet(self, data, addr):
        """Process received TCP packet and check for reordering."""
        try:
            # Check packet length
            if len(data) < 8:
                logging.warning(f"[{self.ip}]: Received malformed packet (too short): {len(data)} bytes")
                return

            # Extract flow_id and seq from the first 8 bytes.
            # Using struct.unpack for clarity.
            flow_id, seq = struct.unpack("!II", data[0:8])
            
            arrival_time = time.time()
            src_ip = addr[0]
            dst_ip = self.ip
            dport = self.port

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

            #logging.debug(f"Received packet: flow_id={flow_id}, seq={seq} from {src_ip} to {dst_ip}:{dport}. Reorder flag: {reorder_flag}")

            # Log packet details to CSV
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([arrival_time, src_ip, dst_ip, dport, flow_id, seq, reorder_flag])
                
        except Exception as e:
            logging.error(f"[{self.ip}]: Error processing packet from {addr[0]}:{addr[1]}: {e}")
            logging.error(traceback.format_exc())


class DataCollectionServer(BaseServer):
    """
    Server that handles both background UDP flows and responds to burst requests.
    """
    def __init__(self, ip=None, port=12345, exp_id='', log_file='receiver_log.csv', burst_reply_size=40000):
        super().__init__(port, ip, exp_id=exp_id)
        self.log_file = log_file
        self.highest_seq = {}  # Store highest sequence number seen per flow
        self.burst_reply_size = burst_reply_size  # Size of response for burst requests
        
        # Initialize the CSV log file
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "src_ip", "dst_ip", "port", "flow_id", "seq", "packet_size", "packet_type", "reordering_flag"])
        
        logging.info(f"[{self.ip}]: Mixed Collection Server initialized with log file: {self.log_file}")
    
    def start(self):
        """Start the UDP server for packet collection."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', self.port))
            logging.info(f"[{self.ip}]: Mixed Collection server listening on UDP port {self.port}")
            
            try:
                while True:
                    try:
                        data, addr = s.recvfrom(4096)
                        # Process received packet (background or burst request)
                        self.process_packet(data, addr, s)
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logging.error(f"[{self.ip}]: Error receiving packet: {e}")
                        logging.error(traceback.format_exc())
            except KeyboardInterrupt:
                logging.info(f"[{self.ip}]: Server shutting down gracefully.")
    
    def process_packet(self, data, addr, socket_conn):
        """Process received UDP packet and check for type."""
        try:
            # Check packet length
            if len(data) < 9:  # flow_id(4) + seq(4) + type(1)
                logging.warning(f"[{self.ip}]: Received malformed packet (too short): {len(data)} bytes")
                return
            
            # Extract flow_id, seq, and packet type from the header
            flow_id, seq = struct.unpack("!II", data[0:8])
            packet_type = ord(data[8:9])  # Convert byte to integer value properly

            packet_size = len(data) + 28  # 28 bytes for UDP/IP headers
            logging.debug(f"Received packet: type={packet_type}, size={packet_size}, flow={flow_id}, seq={seq}")
            
            arrival_time = time.time()
            src_ip = addr[0]
            dst_ip = self.ip
            dport = self.port
            
            # Define a flow key (using src, dst, port, flow_id)
            key = (src_ip, dst_ip, dport, flow_id)
            
            # Check for packet reordering (only for background flows)
            reorder_flag = 0
            if packet_type == 0:  # Background packet
                if key in self.highest_seq:
                    if seq < self.highest_seq[key]:
                        reorder_flag = 1  # Out-of-order packet
                    else:
                        self.highest_seq[key] = seq
                else:
                    self.highest_seq[key] = seq
            
            # Log packet details to CSV (only background packets)
            if packet_type == 0:
                with open(self.log_file, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([arrival_time, src_ip, dst_ip, dport, flow_id, seq, packet_size, packet_type, reorder_flag])
            else: # type = 1
                self._send_burst_response(flow_id, seq, addr, socket_conn)
                
        except Exception as e:
            logging.error(f"[{self.ip}]: Error processing packet from {addr[0]}:{addr[1]}: {e}")
            logging.error(traceback.format_exc())
    
    def _send_burst_response(self, flow_id, req_seq, addr, socket_conn):
        """Send a response to a burst request."""
        try:
            # Create response packet with same flow ID but different type
            flow_id_bytes = flow_id.to_bytes(4, byteorder='big')
            seq_bytes = req_seq.to_bytes(4, byteorder='big')
            type_byte = b'\x01'  # Burst traffic type (1 for both requests and responses)
            
            # Create a response packet of requested size
            payload_size = self.burst_reply_size - 9  # flow_id(4) + seq(4) + type(1)
            payload = b'R' * payload_size
            
            response_packet = flow_id_bytes + seq_bytes + type_byte + payload
            
            # Send the response back to the client
            socket_conn.sendto(response_packet, addr)
            logging.debug(f"Sent burst response for flow {flow_id} to {addr[0]}:{addr[1]}")
            
        except Exception as e:
            logging.error(f"[{self.ip}]: Error sending burst response to {addr[0]}:{addr[1]}: {e}")
            logging.error(traceback.format_exc())


class BackgroundTcpServer(BaseServer):
    """Server that handles background TCP traffic streams."""
    def __init__(self, ip=None, port=12345, exp_id='', capture_pcap=True):
        super().__init__(port, ip, exp_id=exp_id)
        self.log_file = os.path.join('tmp', exp_id, f"bg_server_{self.ip}_{self.port}.csv")
        self.capture_pcap = capture_pcap
        
        # Initialize the CSV log file
        with open(self.log_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "src_ip", "dst_ip", "port", "bytes_received"])
        
        logging.info(f"[{self.ip}]: Background TCP Server initialized with log file: {self.log_file}")
    
    def start(self):
        """Start the TCP server for background traffic collection."""
        try:
            # Start packet capture if enabled
            if self.capture_pcap and self.exp_id:
                pcap_file = f"tmp/{self.exp_id}/bg_server_{self.ip}_{self.port}.pcap"
                self.start_packet_capture(pcap_file)
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Enable TCP_NODELAY to prevent buffering
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Set MSS (Maximum Segment Size) to influence packet size
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)  # Common datacenter MSS
                
                # Limit the send buffer size to prevent large bursts
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)  # Moderate buffer size
                
                # Disable delayed ACKs
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                
                s.bind(('0.0.0.0', self.port))
                s.listen(10)  # Allow up to 10 pending connections
                logging.info(f"[{self.ip}]: Background TCP server listening on port {self.port}")
                
                while self.running:
                    try:
                        conn, addr = s.accept()
                        # Enable TCP_NODELAY for the connection
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        
                        # Set MSS for this connection
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)
                        
                        # Set send buffer size for this connection
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
                        
                        # Disable delayed ACKs for this connection
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                        
                        # Handle each connection in a separate thread
                        client_thread = threading.Thread(
                            target=self.handle_connection, 
                            args=(conn, addr), 
                            daemon=True
                        )
                        client_thread.start()
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logging.error(f"[{self.ip}]: Error accepting connection: {e}")
                        logging.error(traceback.format_exc())
        except KeyboardInterrupt:
            logging.info(f"[{self.ip}]: Background TCP server shutting down gracefully.")
        except Exception as e:
            logging.error(f"[{self.ip}]: Error in server: {e}")
            logging.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def handle_connection(self, conn, addr):
        """Handle a single TCP connection for background traffic."""
        src_ip = addr[0]
        dst_ip = self.ip
        port = self.port
        total_bytes = 0
        
        try:
            # Set socket timeout
            conn.settimeout(5)
            
            # Start receiving data
            arrival_time = time.time()
            while True:
                data = conn.recv(4096)  # Receive in chunks
                if not data:
                    break
                total_bytes += len(data)
            
            # Log connection details to CSV
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([arrival_time, src_ip, dst_ip, port, total_bytes])
                
            logging.debug(f"Completed background TCP connection from {src_ip}: {total_bytes} bytes")
                
        except Exception as e:
            logging.error(f"[{self.ip}]: Error handling connection from {src_ip}: {e}")
        finally:
            conn.close()


class BurstyTcpServer(BaseServer):
    """Server that handles bursty TCP request/response traffic."""
    def __init__(self, ip=None, port=12346, exp_id='', burst_reply_size=4000, capture_pcap=True):
        super().__init__(port, ip, exp_id=exp_id)
        self.burst_reply_size = burst_reply_size
        self.capture_pcap = capture_pcap
        
        logging.info(f"[{self.ip}]: Bursty TCP Server initialized with reply size: {self.burst_reply_size}")
    
    def start(self):
        """Start the TCP server for bursty request/response traffic."""
        try:
            # Start packet capture if enabled
            if self.capture_pcap and self.exp_id:
                pcap_file = f"tmp/{self.exp_id}/bursty_server_{self.ip}_{self.port}.pcap"
                self.start_packet_capture(pcap_file)
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Enable TCP_NODELAY to prevent buffering
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                
                # Set MSS (Maximum Segment Size) to influence packet size
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)  # Common datacenter MSS
                
                # Limit the send buffer size to prevent large bursts
                s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)  # Moderate buffer size
                
                # Disable delayed ACKs
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                
                s.bind(('0.0.0.0', self.port))
                s.listen(10)  # Allow up to 10 pending connections
                logging.info(f"[{self.ip}]: Bursty TCP server listening on port {self.port}")
                
                while self.running:
                    try:
                        conn, addr = s.accept()
                        # Enable TCP_NODELAY for the connection
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        
                        # Set MSS for this connection
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)
                        
                        # Set send buffer size for this connection
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
                        
                        # Disable delayed ACKs for this connection
                        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                        
                        # Handle each connection in a separate thread
                        client_thread = threading.Thread(
                            target=self.handle_burst_request, 
                            args=(conn, addr), 
                            daemon=True
                        )
                        client_thread.start()
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        logging.error(f"[{self.ip}]: Error accepting connection: {e}")
                        logging.error(traceback.format_exc())
        except KeyboardInterrupt:
            logging.info(f"[{self.ip}]: Bursty TCP server shutting down gracefully.")
        except Exception as e:
            logging.error(f"[{self.ip}]: Error in server: {e}")
            logging.error(traceback.format_exc())
        finally:
            self.cleanup()
    
    def handle_burst_request(self, conn, addr):
        """Handle a burst request - send large response and let TCP handle segmentation."""
        try:
            # Receive any request data (but don't need it)
            conn.recv(1024)

            # Send burst response - TCP will automatically segment based on network parameters
            response = b'B' * self.burst_reply_size
            conn.sendall(response)
            
            logging.debug(f"Sent burst response of {self.burst_reply_size} bytes to {addr[0]}")
            
        except Exception as e:
            logging.error(f"[{self.ip}]: Error handling burst request from {addr[0]}: {e}")
        finally:
            conn.close()
