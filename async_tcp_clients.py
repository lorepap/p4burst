import asyncio
import socket
import time
import random
import logging
import os
import csv
import subprocess
import shutil
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)

class AsyncBaseClient(ABC):
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
        sock.setsockopt(socket.IPPROTO_TCP,
                        socket.TCP_CONGESTION,
                        self.congestion_control.encode())
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1460)
        sock.setsockopt(socket.SOL_SOCKET,
                        socket.SO_SNDBUF,
                        256 * 1024)
        sock.setsockopt(socket.IPPROTO_TCP,
                        socket.TCP_QUICKACK,
                        1)

    def _start_pcap(self):
        if not (self.capture_pcap and self.exp_id):
            return
        cap = f"tmp/{self.exp_id}/{self.__class__.__name__.lower()}_{self.ip}_{self.port}.pcap"
        os.makedirs(os.path.dirname(cap), exist_ok=True)

        # Verifica che tcpdump esista
        tcpdump_bin = shutil.which('tcpdump')
        if not tcpdump_bin:
            logging.error("tcpdump non trovato in PATH, niente pcap")
            return

        cmd = [
            tcpdump_bin,
            '-i', 'any',
            '-w', cap,
            'port', str(self.port),
            '-s', '0', '-U', '-W', '1', '-C', '1000'
        ]

        logging.info(f"Starting pcap {cap} with cmd: {' '.join(cmd)}")
        self.tcpdump = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,    # non serve stdout
            stderr=subprocess.PIPE,       # catturiamo stderr
            preexec_fn=os.setpgrp
        )

        # aspetta un attimo e controlla se è già uscito
        time.sleep(0.1)
        ret = self.tcpdump.poll()
        if ret is not None and ret != 0:
            err = self.tcpdump.stderr.read().decode(errors='ignore').strip()
            logging.error(f"tcpdump non è partito (code={ret}): {err}")
            self.tcpdump = None

    def _stop_pcap(self):
        if not self.tcpdump:
            return
        self.tcpdump.terminate()
        self.tcpdump.wait(timeout=3)
        if self.log:
            logging.info("Stopped pcap")
        self.tcpdump = None

    @abstractmethod
    async def start(self):
        pass

class AsyncBackgroundTcpClient(AsyncBaseClient):
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
        log_flush_interval=1.0,
        flow_size=1000,
        flow_iat=0.1
    ):
        super().__init__(
            server_ips,
            port,
            congestion_control,
            exp_id,
            capture_pcap,
            log,
            duration,
            log_buffer_size,
            log_flush_interval
        )
        self.flow_size = flow_size
        self.flow_iat = flow_iat
        self.log_file = f"tmp/{self.exp_id}/bg_client_{self.ip}_{self.port}.csv"

    async def _send_flow(self, target, log_buf, last_flush):
        flow_id = f"{self.ip.replace('.', '')}_{target.replace('.', '')}"
        start = time.time()
        reader, writer = await asyncio.open_connection(host=target, port=self.port)
        sock = writer.get_extra_info('socket')
        if sock:
            self._apply_socket_options(sock)
        writer.write(b'X' * self.flow_size)
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        end = time.time()
        fct = end - start
        log_buf.append([
            flow_id, start, end, fct,
            self.ip, target, self.flow_size,
            self.congestion_control
        ])
        if len(log_buf) >= self.log_buffer_size or time.time() - last_flush >= self.log_flush_interval:
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerows(log_buf)
            log_buf.clear()
            last_flush = time.time()
        return last_flush

    async def start(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['flow_id','start','end','fct','src_ip','dst_ip','bytes','cc']
            )
        self._start_pcap()
        deadline = time.time() + self.duration
        log_buf = []
        last_flush = time.time()
        while time.time() < deadline:
            target = random.choice([ip for ip in self.server_ips if ip not in self.ip])
            last_flush = await self._send_flow(target, log_buf, last_flush)
            await asyncio.sleep(self.flow_iat)
        if log_buf:
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerows(log_buf)
        if self.log:
            logging.info("Background client done")
        self._stop_pcap()

class AsyncBurstyTcpClient(AsyncBaseClient):
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
        log_flush_interval=1.0,
        burst_interval=1.0,
        burst_servers=None
    ):
        super().__init__(
            server_ips,
            port,
            congestion_control,
            exp_id,
            capture_pcap,
            log,
            duration,
            log_buffer_size,
            log_flush_interval
        )
        self.burst_interval = burst_interval
        self.burst_servers = burst_servers if burst_servers is not None else len(self.server_ips)
        self.log_file = f"tmp/{self.exp_id}/bursty_client_{self.ip}_{self.port}.csv"

    async def _send_query(self, target):
        start = time.time()
        reader, writer = await asyncio.open_connection(host=target, port=self.port)
        sock = writer.get_extra_info('socket')
        if sock:
            self._apply_socket_options(sock)
        writer.write(b'REQ')
        await writer.drain()
        total = 0
        while True:
            data = await reader.read(4096)
            if not data:
                break
            total += len(data)
        writer.close()
        await writer.wait_closed()
        return total, time.time() - start, target

    async def start(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['burst_id','start','qct','n','bytes','cc','slow','slow_t']
            )
        self._start_pcap()
        burst_id = 0
        deadline = time.time() + self.duration
        while time.time() < deadline:
            servers = random.sample(self.server_ips, self.burst_servers)
            t0 = time.time()
            tasks = [self._send_query(s) for s in servers]
            results = await asyncio.gather(*tasks)
            qct = time.time() - t0
            total = sum(r[0] for r in results)
            slow = max(results, key=lambda r: r[1])
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerow(
                    [f"burst_{burst_id}", t0, qct, len(servers), total,
                     self.congestion_control, slow[2], slow[1]]
                )
            burst_id += 1
            await asyncio.sleep(self.burst_interval)
        if self.log:
            logging.info("Bursty client done")
        self._stop_pcap()
