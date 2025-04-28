import asyncio
import socket
import time
import logging
import traceback
import csv
import os
import subprocess
import signal
import sys
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)

class AsyncBaseServer(ABC):
    def __init__(self, port=12345, ip=None, congestion_control='cubic',
                 exp_id='', log=True, capture_pcap=True, backlog=1024):
        self.port = int(port)
        self.ip = ip or (_ for _ in ()).throw(ValueError("IP address must be specified"))
        self.congestion_control = congestion_control
        self.exp_id = exp_id
        self.log = log
        self.capture_pcap = capture_pcap
        self.backlog = backlog
        self.tcpdump_process = None
        # spostiamo la creazione dell'Event nel loop corretto
        self.shutdown_event = None

    def _signal_handler(self):
        if self.log:
            logging.info(f"[{self.ip}]: Received shutdown signal")
        self.shutdown_event.set()
        if self.capture_pcap:
            self.stop_packet_capture()

    def start(self):
        """Start the asyncio server."""
        asyncio.run(self._run_server())

    async def _run_server(self):
        # creare l'Event sul loop corrente
        self.shutdown_event = asyncio.Event()

        # Setup packet capture
        if self.capture_pcap:
            self.start_packet_capture()

        # Create and configure listening socket
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION,
                             self.congestion_control.encode())
        srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        srv_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
        srv_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)
        srv_sock.bind(('0.0.0.0', self.port))
        srv_sock.listen(self.backlog)
        srv_sock.setblocking(False)

        loop = asyncio.get_running_loop()
        # Register signals for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

        '''
        server = await loop.create_server(
            lambda: AsyncConnectionHandler(self),
            sock=srv_sock)
        '''
        
        server = await asyncio.start_server(
            self.handle_request,
            sock=srv_sock,
            #backlog=self.backlog,
        )

        if self.log:
            logging.info(f"[{self.ip}]: Async server listening on port {self.port} "
                         f"(CC={self.congestion_control}, RCVBUF=256KiB, backlog={self.backlog})")

        # Wait for shutdown
        await self.shutdown_event.wait()

        # Close server
        server.close()
        await server.wait_closed()
        if self.log:
            logging.info(f"[{self.ip}]: Server shut down")

    @abstractmethod
    async def handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Override in subclasses to handle a single connection."""
        pass

    def start_packet_capture(self):
        """Start tcpdump packet capture to a pcap file."""
        if not self.exp_id:
            return
        server_type = self.__class__.__name__.lower()
        capture_file = f"tmp/{self.exp_id}/{server_type}_{self.ip}_{self.port}.pcap"
        os.makedirs(os.path.dirname(capture_file), exist_ok=True)
        try:
            if self.log:
                logging.info(f"[{self.ip}]: Starting packet capture to {capture_file}")
            cmd = [
                "sudo", "-n", "tcpdump", "-i", "any",
                "-w", capture_file,
                f"port {self.port}", "-s", "0", "-U", "-W", "1", "-C", "1000"
            ]
            self.tcpdump_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=os.setpgrp
            )
        except Exception as e:
            logging.error(f"[{self.ip}]: Failed to start packet capture: {e}")
            self.tcpdump_process = None

    def stop_packet_capture(self):
        """Stop the tcpdump packet capture process."""
        if not self.tcpdump_process:
            return
        try:
            if self.log:
                logging.info(f"[{self.ip}]: Stopping packet capture")
            self.tcpdump_process.terminate()
            try:
                self.tcpdump_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.tcpdump_process.send_signal(signal.SIGINT)
                try:
                    self.tcpdump_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.tcpdump_process.kill()
                    self.tcpdump_process.wait(timeout=1)
            if self.log:
                logging.info(f"[{self.ip}]: Packet capture stopped")
        except Exception as e:
            logging.error(f"[{self.ip}]: Error stopping packet capture: {e}")
        finally:
            self.tcpdump_process = None

class AsyncConnectionHandler(asyncio.Protocol):
    def __init__(self, server: AsyncBaseServer):
        self.server = server
        self.transport = None

    def connection_made(self, transport: asyncio.BaseTransport):
        self.transport = transport
        peer = transport.get_extra_info('peername')
        if self.server.log:
            logging.info(f"[{self.server.ip}]: Connection from {peer}")

    def data_received(self, data: bytes):
        # Not used; we handle via streams in subclasses
        pass

    def eof_received(self):
        pass

    def connection_lost(self, exc):
        # Called when connection is closed or error
        pass

class AsyncBackgroundTcpServer(AsyncBaseServer):
    """Async server for background TCP streams."""
    async def handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        src_ip = addr[0]
        total_bytes = 0
        arrival = time.time()
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                total_bytes += len(chunk)
            if self.log:
                logging.debug(f"[{self.ip}]: Background from {src_ip}: {total_bytes} bytes")
        except Exception as e:
            if self.log:
                logging.error(f"[{self.ip}]: Error background conn: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

class AsyncBurstyTcpServer(AsyncBaseServer):
    """Async server for bursty TCP request/response traffic."""
    def __init__(self, *args, burst_reply_size=4000, **kwargs):
        super().__init__(*args, **kwargs)
        self.burst_reply_size = burst_reply_size

    async def handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        try:
            # Read request
            _ = await reader.read(1024)
            # Send burst
            writer.write(b'B' * self.burst_reply_size)
            await writer.drain()
            if self.log:
                logging.debug(f"[{self.ip}]: Sent burst of {self.burst_reply_size} bytes to {addr[0]}")
        except Exception as e:
            if self.log:
                logging.error(f"[{self.ip}]: Error bursty conn: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
