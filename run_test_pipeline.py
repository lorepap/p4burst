#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(description="RL pipeline: Data collection, DQN training, and Decision Tree distillation.")

# Parameters
parser.add_argument('--log_dir', type=str, default='tmp/0000-deflection', help='Directory for logs and CSV datasets.')
parser.add_argument('--n_pkts', type=int, default=100, help='Number of packets to send during data collection.')
parser.add_argument('--n_flows', type=int, default=1, help='Number of flows to simulate during data collection.')
parser.add_argument('--n_clients', type=int, default=1, help='Number of clients to simulate during data collection.')
parser.add_argument('--interval', type=float, default=0.001, help='Inter-arrival interval between packets.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for DQN training.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DQN training.')
parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the DQN model.')
parser.add_argument('--debug', action='store_true', help='Run with debug defaults.')

args = parser.parse_args()

# Debug defaults
if args.debug:
    n_packets = 10
    interval = 0.0001
    epochs = 300
    n_flows = 1
    log_dir = 'tmp/0000-deflection'
else:
    n_packets = args.n_pkts
    interval = args.interval
    n_flows = args.n_flows
    n_clients = args.n_clients
    epochs = args.epochs
    log_dir = args.log_dir

# Ensure directories exist
os.makedirs(log_dir, exist_ok=True)
os.makedirs(args.model_dir, exist_ok=True)

# 1. Data collection
os.system(f"sudo -E python test_data_collection.py --n_pkts {n_packets} --interval {interval} --n_flows {n_flows} --n_clients {n_clients}")

# 2. DQN offline training
os.system(f"sudo -E python rl_agent.py --csv {log_dir}/final_rl_dataset.csv --epochs {epochs} --batch_size {args.batch_size} --model_dir {args.model_dir}")

# 3. Decision Tree distillation
os.system(f"python dt.py --csv {log_dir}/final_rl_dataset.csv --model {args.model_dir}/dqn_model.h5")

print("RL pipeline execution completed.")