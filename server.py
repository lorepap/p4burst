#!/usr/bin/env python3
import argparse
import logging
import os
from async_tcp_servers import TcpServer

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Unified TCP server")
    parser.add_argument('--exp_id', required=True, help="Experiment ID")
    parser.add_argument('--host_ip', required=True, help="Server IP address")
    parser.add_argument('--port', type=int, default=12345, help="TCP port to listen on")
    parser.add_argument('--reply_size', type=int, default=1000, help="Bytes to reply per request")
    parser.add_argument('--disable_logging', action='store_true')
    parser.add_argument('--disable_pcap', action='store_true')
    args = parser.parse_args()

    if not args.disable_logging:
        os.makedirs(f"tmp/{args.exp_id}", exist_ok=True)

    server = TcpServer(
        ip=args.host_ip,
        port=args.port,
        exp_id=args.exp_id,
        reply_size=args.reply_size,
        capture_pcap=not args.disable_pcap,
        log=not args.disable_logging
    )
    server.start()

if __name__ == '__main__':
    main()
    