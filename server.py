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

class BaseServer(ABC):
    def __init__(self, port=12345, ip=None, congestion_control='cubic', exp_id=''):
        self.port = port
        self.congestion_control = congestion_control
        self.exp_id = exp_id
        # self.ip = self.get_host_ip()
        if ip:
            self.ip = ip 
        else: raise ValueError("IP address must be specified")

        if self.exp_id:
            self.flowtracker = FlowMetricsManager(self.exp_id)

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
                while True:
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
            except Exception as e:
                logging.error(f"[{self.ip}]: Error accepting connection: {e}")
                logging.error(traceback.format_exc())


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
        super().__init__(port, ip, cong_control, exp_id)

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


class IperfServer(BaseServer):
    def __init__(self, port=5201):
        self.port = port

    def start(self):
        cmd = f'iperf3 -s &'
        os.system(cmd)

