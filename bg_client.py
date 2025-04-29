#!/usr/bin/env python3
import argparse
import asyncio
import logging
from async_tcp_clients import AsyncBackgroundTcpClient
import os

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Background TCP client (multiple flows)")
    parser.add_argument('--exp_id', required=True)
    parser.add_argument('--server_ips', nargs='+', required=True)
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--reply_size', type=int, default=1000, help="Bytes per flow reply")
    parser.add_argument('--interval', type=float, default=0.1, help="Inter-arrival time between flows")
    parser.add_argument('--duration', type=int, default=60, help="Duration in seconds")
    parser.add_argument('--cc', default='cubic', help="Congestion control algorithm")
    parser.add_argument('--disable_logging', action='store_true')
    parser.add_argument('--disable_pcap', action='store_true')
    args = parser.parse_args()

    logging.info("Starting background client")
    client = AsyncBackgroundTcpClient(
        server_ips=args.server_ips,
        port=args.port,
        exp_id=args.exp_id,
        capture_pcap=not args.disable_pcap,
        log=not args.disable_logging,
        duration=args.duration,
        flow_size=args.reply_size,
        flow_iat=args.interval,
        congestion_control=args.cc
    )
    asyncio.run(client.start())

if __name__ == '__main__':
    main()
    