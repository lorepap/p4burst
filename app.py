import argparse
import asyncio
import logging
import os
import configparser
from async_tcp_clients import AsyncBackgroundTcpClient, AsyncBurstyTcpClient
from async_tcp_servers import AsyncBackgroundTcpServer, AsyncBurstyTcpServer
import sys

def lower_priority(nice_level: int = 10):
    try:
        os.nice(nice_level)
        logging.info(f"[init] nice level set to +{nice_level}")
    except Exception as e:
        logging.warning(f"[init] could not set nice level: {e}")

def derive_core_from_ip(ip: str, ncores: int) -> int:
    """
    Deriva un core (da 1 a ncores-1) basandosi 
    sull'ultimo ottetto dell'IP per evitare il core 0.
    """
    try:
        last = int(ip.split('.')[-1])
        # mod (ncores-1) +1 assicura core in [1, ncores-1]
        return (last % (ncores - 1)) + 1
    except Exception:
        return 1

def pin_to_core(core_id: int):
    pid = os.getpid()
    try:
        os.sched_setaffinity(pid, {core_id})
        logging.info(f"[init] pinned process to CPU core {core_id}")
    except Exception as e:
        logging.warning(f"[init] could not set CPU affinity: {e}")

class App:
    def __init__(self, args):
        self.mode = args.mode
        self.client = None
        self.server = None
        self.config = self.load_config(args.config_file)
        # prepare experiment directory and logging
        if not args.disable_logging:
            exp_dir = os.path.join('tmp', args.exp_id)
            os.makedirs(exp_dir, exist_ok=True)
            self.setup_logging(log_file=os.path.join(exp_dir, 'app.log'))
        self.args = args

    @staticmethod
    def setup_logging(log_file='tmp/app.log'):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logger.handlers = []
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    @staticmethod
    def load_config(config_file):
        config = configparser.ConfigParser()
        config.read(config_file)
        return config

    def run(self):
        if self.mode == 'server':
            self.run_server()
        else:
            # client mode
            asyncio.run(self.run_client())

    def run_server(self):
        if self.args.traffic_type == 'background':
            self.server = AsyncBackgroundTcpServer(
                ip=self.args.host_ip,
                port=self.args.port,
                exp_id=self.args.exp_id,
                capture_pcap=not self.args.disable_pcap,
                log=not self.args.disable_logging
            )
        else:
            self.server = AsyncBurstyTcpServer(
                ip=self.args.host_ip,
                port=self.args.port,
                exp_id=self.args.exp_id,
                burst_reply_size=self.args.burst_reply_size,
                capture_pcap=not self.args.disable_pcap,
                log=not self.args.disable_logging
            )
        self.server.start()

    async def run_client(self):
        if self.args.traffic_type == 'background':
            self.client = AsyncBackgroundTcpClient(
                server_ips=self.args.server_ips,
                port=self.args.port,
                exp_id=self.args.exp_id,
                congestion_control=self.args.congestion_control,
                capture_pcap=not self.args.disable_pcap,
                log=not self.args.disable_logging,
                duration=self.args.duration,
                flow_iat=self.args.bg_flow_iat,
                flow_size=self.args.flow_size
            )
        else:
            self.client = AsyncBurstyTcpClient(
                server_ips=self.args.server_ips,
                port=self.args.port,
                exp_id=self.args.exp_id,
                congestion_control=self.args.congestion_control,
                capture_pcap=not self.args.disable_pcap,
                log=not self.args.disable_logging,
                duration=self.args.duration,
                burst_interval=self.args.burst_interval,
                burst_servers=self.args.burst_servers
            )
        await self.client.start()


def main():
    parser = argparse.ArgumentParser(description="Mixed TCP traffic app (async)")
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--exp_id', required=True, help="Experiment ID")
    parser.add_argument('--config_file', default='config.ini', help="Path to config file")
    parser.add_argument('--host_ip', required=True, help="This node's IP address")
    parser.add_argument('--server_ips', nargs='+', help="List of server IPs (client mode)")
    parser.add_argument('--traffic_type', choices=['background', 'burst'], required=True)
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--disable_logging', action='store_true')
    parser.add_argument('--disable_pcap', action='store_true')
    parser.add_argument('--congestion_control', default='cubic')
    parser.add_argument('--duration', type=int, default=60)
    # background specific
    parser.add_argument('--bg_flow_iat', type=float, default=0.1)
    parser.add_argument('--flow_size', type=int, default=1000)
    # burst specific
    parser.add_argument('--burst_interval', type=float, default=1.0)
    parser.add_argument('--burst_servers', type=int, default=2)
    parser.add_argument('--burst_reply_size', type=int, default=4000)
    parser.add_argument('--core', type=int, default=1, help="CPU core ID to pin to")
    
    #ncores = os.cpu_count() or 1
    #logging.info(f"[init] total CPU cores available: {ncores}")

    # Deriva un core dall'IP
    #core = derive_core_from_ip(args.host_ip, ncores)
    #logging.info(f"[init] derived CPU core from IP {args.host_ip}: {core}")

    # Fissa affinità
    #pin_to_core(core)
    #lower_priority(15)

    args = parser.parse_args()
    pin_to_core(args.core)

    app = App(args)
    app.run()

if __name__ == '__main__':
    main()
