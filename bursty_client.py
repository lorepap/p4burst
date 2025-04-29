#!/usr/bin/env python3
import argparse
import asyncio
import logging
from async_tcp_clients import AsyncBurstyTcpClient

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Bursty TCP client (parallel queries)")
    parser.add_argument('--exp_id', required=True)
    parser.add_argument('--server_ips', nargs='+', required=True)
    parser.add_argument('--port', type=int, default=12345)
    parser.add_argument('--reply_size', type=int, default=1000, help="Bytes per query reply")
    parser.add_argument('--interval', type=float, default=1.0, help="Interval between bursts")
    parser.add_argument('--burst_servers', type=int, default=1, help="Number of parallel servers per burst")
    parser.add_argument('--duration', type=int, default=60, help="Duration in seconds")
    parser.add_argument('--cc', default='cubic', help="Congestion control algorithm")
    parser.add_argument('--disable_logging', action='store_true')
    parser.add_argument('--disable_pcap', action='store_true')
    args = parser.parse_args()

    logging.info("Starting bursty client")
    client = AsyncBurstyTcpClient(
        server_ips=args.server_ips,
        port=args.port,
        exp_id=args.exp_id,
        capture_pcap=not args.disable_pcap,
        log=not args.disable_logging,
        duration=args.duration,
        burst_interval=args.interval,
        burst_servers=args.burst_servers,
        burst_reply_size=args.reply_size,
        congestion_control=args.cc
    )
    asyncio.run(client.start())

if __name__ == '__main__':
    main()
