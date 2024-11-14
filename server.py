import socket
import time
from abc import ABC, abstractmethod
import logging
import traceback
import math
import csv
from metrics import FlowMetricsManager
import threading

class BaseServer(ABC):
    def __init__(self, port=12345):
        self.port = port
        self.ip = self.get_host_ip()
        self.flowtracker = FlowMetricsManager()

    @abstractmethod
    def handle_request(self, conn):
        pass

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen()
            logging.info(f"[{self.ip}]: Server listening on port {self.port}")
            while True:
                try:
                    conn, addr = s.accept()
                    logging.info(f"[{self.ip}]: Accepted connection from {addr}")
                    with conn:
                        print(f"Connected by {addr}")
                        self.handle_request(conn, addr)
                except Exception as e:
                    logging.error(f"[{self.ip}]: Error accepting connection: {e}")
                    logging.error(traceback.format_exc())

    def get_host_ip(self):
        # try:
            # Create a socket to determine the outgoing IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Doesn't actually send any data
        ip = s.getsockname()[0]
        s.close()
        return ip
        # except Exception:
        #     traceback.print_exc()
        #     sys.exit(1) 


class BurstyServer(BaseServer):
    def __init__(self, reply_size=40000, port=12345, mtu=1500, iat=0.001):
        super().__init__(port)
        self.reply_size = reply_size
        self.mtu = mtu
        self.iat = iat  # Inter-Arrival Time in seconds

    def handle_request(self, conn, addr):
        data = conn.recv(1024)
        if data:
            logging.info(f"[{self.ip}]: Received request from {addr[0]}:{addr[1]}")
            
            # Calculate number of packets
            payload_size = self.mtu - 40  # Assuming 20 bytes for IP header and 20 for TCP header
            num_packets = math.ceil(self.reply_size / payload_size)
            
            bytes_sent = 0
            for i in range(num_packets):
                if i == num_packets - 1:  # Last packet
                    packet_size = self.reply_size - bytes_sent
                else:
                    packet_size = payload_size
                
                packet = b'X' * packet_size
                conn.sendall(packet)
                bytes_sent += packet_size
                
                logging.info(f"[{self.ip}]: Sent packet {i+1}/{num_packets} of {packet_size} bytes to {addr[0]}:{addr[1]}")
                
                if i < num_packets - 1:  # Don't sleep after the last packet
                    time.sleep(self.iat)

            logging.info(f"[{self.ip}]: Completed sending response of {bytes_sent} bytes in {num_packets} packets to {addr[0]}:{addr[1]}")


class BackgroundServer(BaseServer):
    def __init__(self, port=12345):
        super().__init__(port)

    def start(self):
        """Start the server with multi-threading support."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen()
            logging.info(f"[{self.ip}]: Server listening on port {self.port}")

            while True:
                conn, addr = s.accept()
                logging.info(f"[{self.ip}]: Accepted connection from {addr}")

                # Start a new thread for each client connection
                client_thread = threading.Thread(target=self.handle_request, args=(conn, addr))
                client_thread.daemon = True  # Daemon thread will exit when main program exits
                client_thread.start()

    def handle_request(self, conn, addr):
        """Handle incoming data, compute metrics, and send acknowledgment."""
        total_bytes_received = 0
        flow_id = None  # Assuming each connection has a unique flow_id assigned by the client
        try:
            data = conn.recv(4096)
            if data:
                flow_id = int(data[:8].decode('utf-8'))
                total_bytes_received += len(data) - 8
                logging.info(f"Started Flow ID: {flow_id} | Bytes received: {total_bytes_received}")
            while data:
                data = conn.recv(4096)
                if not data:
                    break
                total_bytes_received += len(data)

            # Mark the flow as complete in FlowMetricsManager
            if flow_id is not None:
                FlowMetricsManager().complete_flow(flow_id)
                logging.info(f"Flow {flow_id} completed. Total bytes received: {total_bytes_received}")
                logging.info(f"Metrics: {FlowMetricsManager().get_metrics()}")
                print("Metrics:", FlowMetricsManager().get_metrics())

        except Exception as e:
            logging.error(f"[{self.ip}]: Error receiving data from {addr[0]}:{addr[1]}: {e}")

    def get_metrics(self):
        """Retrieve collected flow metrics."""
        return FlowMetricsManager().get_metrics()