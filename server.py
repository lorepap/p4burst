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

class CollectionServer(BaseServer):
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

            logging.debug(f"Received packet: flow_id={flow_id}, seq={seq} from {src_ip} to {dst_ip}:{dport}. Reorder flag: {reorder_flag}")

            # Log packet details to CSV
            with open(self.log_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([arrival_time, src_ip, dst_ip, dport, flow_id, seq, reorder_flag])
                
        except Exception as e:
            logging.error(f"[{self.ip}]: Error processing packet from {addr[0]}:{addr[1]}: {e}")
            logging.error(traceback.format_exc())


class IperfServer(BaseServer):
    def __init__(self, port=5201):
        self.port = port

    def start(self):
        cmd = f'iperf3 -s &'
        os.system(cmd)

