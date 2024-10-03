import argparse
import socket
import time
import random
from abc import ABC, abstractmethod
from client import BurstyClient
from server import BurstyServer
import logging
import os


class App(ABC):
    def __init__(self, mode, log=True):
        self.mode = mode
        self.client = None
        self.server = None
        if log:
            self.setup_logging()

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def setup_logging(log_file='tmp/app.log'):
        os.makedirs('tmp', exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()  # This will also print logs to console
            ])

    def cleanup(self):
        if self.server:
            self.server.stop()
        # remove log files


class BurstyApp(App):
    def __init__(self, mode, reply_size=40000, server_ips=None):
        super().__init__(mode)
        if mode == 'server':
            self.server = BurstyServer(reply_size)
        elif mode == 'client':
            if server_ips is None:
                raise ValueError("server_ips must be provided for client mode")
            self.client = BurstyClient(server_ips, reply_size)
        else:
            raise ValueError("Invalid mode. Choose 'server' or 'client'.")

    def run(self):
        if self.mode == 'server':
            self.server.start()
        elif self.mode == 'client':
            self.client.run()

def main():
    parser = argparse.ArgumentParser(description="Bursty Application")
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--type', choices=['bursty'], required=True)
    parser.add_argument('--reply_size', type=int, default=40000)
    parser.add_argument('--server_ips', nargs='+', required=False, help="List of server IPs (bursty app)")
    args = parser.parse_args()

    # TODO input validation
    app = BurstyApp(args.mode, args.reply_size, args.server_ips)
    app.run()

if __name__ == "__main__":
    main()