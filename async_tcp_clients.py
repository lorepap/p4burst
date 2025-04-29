import asyncio
import socket
import time
import random
import logging
import os
import csv
import subprocess
import shutil

#logging.basicConfig(level=logging.INFO)

class AsyncBaseClient:
    """
    Base client: gestisce opzioni socket, pcap e logging.
    """
    def __init__(
        self,
        server_ips,
        port,
        congestion_control='cubic',
        exp_id='',
        capture_pcap=True,
        log=True,
        duration=None,
        log_buffer_size=1000,
        log_flush_interval=1.0
    ):
        self.server_ips = server_ips if isinstance(server_ips, list) else [server_ips]
        self.port = port
        self.congestion_control = congestion_control
        self.exp_id = exp_id
        self.capture_pcap = capture_pcap
        self.log = log
        self.duration = duration or float('inf')
        self.log_buffer_size = log_buffer_size
        self.log_flush_interval = log_flush_interval
        self.ip = self._get_host_ip() or '0.0.0.0'
        self.tcpdump = None

    def _get_host_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            if self.log:
                logging.warning("Could not determine host IP, defaulting to 0.0.0.0")
            return None

    def _apply_socket_options(self, sock: socket.socket):
        sock.setblocking(False)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, self.congestion_control.encode())
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 256 * 1024)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)

    def _start_pcap(self):
        if not (self.capture_pcap and self.exp_id):
            return
        cap = f"tmp/{self.exp_id}/{self.__class__.__name__.lower()}_{self.ip}_{self.port}.pcap"
        os.makedirs(os.path.dirname(cap), exist_ok=True)
        tcpdump_bin = shutil.which('tcpdump')
        if not tcpdump_bin:
            logging.error("tcpdump non trovato in PATH, niente pcap")
            return
        cmd = [tcpdump_bin, '-i', 'any', '-w', cap, 'port', str(self.port), '-s', '0', '-U', '-W', '1', '-C', '1000']
        logging.info(f"Starting pcap {cap}")
        self.tcpdump = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, preexec_fn=os.setpgrp)
        time.sleep(0.1)
        if self.tcpdump and self.tcpdump.poll() not in (None, 0):
            err = self.tcpdump.stderr.read().decode(errors='ignore').strip()
            logging.error(f"tcpdump non è partito (code={self.tcpdump.returncode}): {err}")
            self.tcpdump = None

    def _stop_pcap(self):
        if not self.tcpdump:
            return
        self.tcpdump.terminate()
        try:
            self.tcpdump.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.tcpdump.kill(); self.tcpdump.wait(timeout=1)
        if self.log:
            logging.info("Stopped pcap")
        self.tcpdump = None

    async def _send_request(self, target: str, expected_bytes: int) -> tuple:
        """
        Invia 'REQ' a target, riceve expected_bytes, ritorna (total_bytes, duration, target).
        """
        start = time.time()
        reader, writer = await asyncio.open_connection(host=target, port=self.port)
        sock = writer.get_extra_info('socket')
        if sock:
            self._apply_socket_options(sock)
        writer.write(b'REQ')
        await writer.drain()
        total = 0
        while total < expected_bytes:
            chunk = await reader.read(expected_bytes - total)
            if not chunk:
                break
            total += len(chunk)
        writer.close()
        await writer.wait_closed()
        duration = time.time() - start
        if self.log:
            logging.info(f"Received {total} bytes from {target} in {duration:.6f}s")
        return total, duration, target

class AsyncBackgroundTcpClient(AsyncBaseClient):
    """
    Client background: invia ripetuti flussi a intervalli `flow_iat`, ricevendo `flow_size` byte.
    """
    def __init__(self, server_ips, port, exp_id='', capture_pcap=True, log=True,
                 duration=None, flow_size=1000, flow_iat=0.1, congestion_control='cubic'):
        super().__init__(server_ips, port, congestion_control, exp_id, capture_pcap, log, duration)
        self.flow_size = flow_size
        self.flow_iat = flow_iat
        self.log_file = f"tmp/{self.exp_id}/bg_client_{self.ip}_{self.port}.csv"

    async def start(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['flow_id','start','end','fct','src_ip','dst_ip','bytes','cc'])
        self._start_pcap()
        deadline = time.time() + self.duration
        flow_count = 0
        while time.time() < deadline:
            target = random.choice([ip for ip in self.server_ips if ip != self.ip])
            flow_id = f"{self.ip.replace('.', '')}_{target.replace('.', '')}_{flow_count}"
            total, dur, _ = await self._send_request(target, self.flow_size)
            end = time.time()
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerow([flow_id, end-dur, end, dur, self.ip, target, total, self.congestion_control])
            flow_count += 1
            await asyncio.sleep(self.flow_iat)
        if self.log:
            logging.info("Background client completed")
        self._stop_pcap()

class AsyncBurstyTcpClient(AsyncBaseClient):
    """
    Client bursty: invia query parallele ogni `burst_interval`, ricevendo `burst_reply_size` byte.
    """
    def __init__(self, server_ips, port, exp_id='', capture_pcap=True, log=True,
                 duration=None, burst_interval=1.0, burst_servers=1, burst_reply_size=4000,
                 congestion_control='cubic'):
        super().__init__(server_ips, port, congestion_control, exp_id, capture_pcap, log, duration)
        self.burst_interval = burst_interval
        self.burst_servers = burst_servers
        self.burst_reply_size = burst_reply_size
        self.log_file = f"tmp/{self.exp_id}/bursty_client_{self.ip}_{self.port}.csv"

    async def start(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['burst_id','start','qct','n','bytes','cc','slow','slow_t'])
        self._start_pcap()
        burst_id = 0
        deadline = time.time() + self.duration
        while time.time() < deadline:
            t0 = time.time()
            targets = random.sample(self.server_ips, self.burst_servers)
            tasks = [self._send_request(t, self.burst_reply_size) for t in targets]
            results = await asyncio.gather(*tasks)
            qct = time.time() - t0
            total = sum(r[0] for r in results)
            slow = max(results, key=lambda x: x[1])
            if self.log:
                times = [f"{r[2]}: {r[1]:.6f}s" for r in results]
                logging.info(f"Burst {burst_id} times: {', '.join(times)}")
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerow([f"burst_{burst_id}", t0, qct, len(results), total,
                                         self.congestion_control, slow[2], slow[1]])
            burst_id += 1
            await asyncio.sleep(self.burst_interval)
        if self.log:
            logging.info("Bursty client completed")
        self._stop_pcap()
