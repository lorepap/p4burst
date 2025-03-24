#!/usr/bin/env python3

"""
Experiment Runner with Predefined Parameters

This script runs multiple data collection experiments with predefined network parameters
to create a comprehensive dataset for training ML models.
"""

import os
import sys
import subprocess
import logging
import time
import itertools
from datetime import datetime
import pandas as pd
import json
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("experiments.log")
    ]
)
logger = logging.getLogger("ExperimentRunner")

class ExperimentRunner:
    def __init__(self, output_dir, config_file=None):
        self.base_dir = output_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create experiment config directory
        self.config_dir = os.path.join(self.base_dir, 'experiment_configs')
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Experiment summary
        self.experiment_summary = []
        self.summary_file = os.path.join(self.base_dir, 'experiment_summary.csv')
        
        # Load or create parameters
        if config_file and os.path.exists(config_file):
            self.param_combinations = self._load_config(config_file)
            logger.info(f"Loaded {len(self.param_combinations)} experiments from config file")
        else:
            self.param_combinations = self._generate_param_combinations()
            logger.info(f"Generated {len(self.param_combinations)} parameter combinations")
            
        # Save experiment configurations
        self._save_configurations()

    def _generate_param_combinations(self):
        """Generate parameter combinations using predefined ranges"""
        
        # Define parameter ranges
        bandwidth_values = [50]  # Mbps
        delay_values = [0.005]  # ms
        burst_server_values = [2, 8, 16]  # Number of servers for bursty traffic
        load_values = [0.25, 0.5, 0.75, 1.0]  # Load percentage (0.25 = 25% load)
        
        # Generate parameter combinations
        combinations = []
        
        # Base packet rate for max load
        base_pps = 1000  # packets per second at full load
        
        experiment_id = 0
        for bw, delay, burst_servers, load in itertools.product(
            bandwidth_values, delay_values, burst_server_values, load_values
        ):
            # Calculate inter-packet interval based on load
            interval = 1 / (base_pps * load)
            
            # Number of background flows increases with load
            #num_flows = max(1, int(load * 5))
            
            # Calculate experiment duration - longer for higher load scenarios
            duration = 5
            
            burst_interval = 0.01
            
            # Create parameter set
            params = {
                'experiment_id': experiment_id,
                'bw': bw,
                'delay': delay,
                'load': load,
                #'num_flows': num_flows,
                'interval': interval,
                'duration': duration,
                'burst_interval': burst_interval,
                'num_packets': 500,
                'n_hosts': 32,  # Fixed parameters
                'n_leaf': 8,
                'n_spine': 4,
                'packet_size': 1472,
                'bursty_reply_size': 4000,
                'n_clients': 16,
                'n_servers': 16,  # Ensure enough servers for burst traffic
                'burst_servers': burst_servers,
                'queue_rate': 100,
                'queue_depth': 30
            }
            
            combinations.append(params)
            experiment_id += 1
            
        return combinations

    def _load_config(self, config_file):
        """Load parameter combinations from a config file"""
        with open(config_file, 'r') as f:
            return json.load(f)

    def _save_configurations(self):
        """Save all experiment configurations to files"""
        # Save all combinations to a single JSON file
        with open(os.path.join(self.config_dir, 'all_experiments.json'), 'w') as f:
            json.dump(self.param_combinations, f, indent=2)
            
        # Save individual experiment configs
        for params in self.param_combinations:
            exp_id = params['experiment_id']
            with open(os.path.join(self.config_dir, f'experiment_{exp_id}.json'), 'w') as f:
                json.dump(params, f, indent=2)

    def run_experiment(self, params):
        """Run a single experiment with the given parameters"""
        exp_id = params['experiment_id']
        
        # Create descriptive experiment ID
        desc_exp_id = f"bw_{params['bw']}_delay_{params['delay']}_load_{int(params['load']*100)}_burst_{params['burst_servers']}_burst_reply_{params['bursty_reply_size']}_bursty_interval_{params['burst_interval']}"
        logger.info(f"Starting experiment {exp_id} with ID: {desc_exp_id}")
        
        # Create the command with all necessary parameters
        cmd = [
            "sudo", "-E", "python3", "collection_runner.py",
            "--exp_id", desc_exp_id,  # Use the descriptive ID
            "--duration", str(params['duration']),
            "--n_hosts", str(params['n_hosts']),
            "--n_leaf", str(params['n_leaf']),
            "--n_spine", str(params['n_spine']),
            "--bw", str(params['bw']),
            "--delay", str(params['delay']),
            "--n_clients", str(params['n_clients']),
            "--n_servers", str(params['n_servers']),
            # "--num_flows", str(params['num_flows']),
            "--interval", str(params['interval']),
            "--packet_size", str(params['packet_size']),
            "--bursty_reply_size", str(params['bursty_reply_size']),
            "--burst_interval", str(params['burst_interval']),
            "--burst_servers", str(params['burst_servers']),
            "--queue_rate", str(params['queue_rate']),
            "--queue_depth", str(params['queue_depth'])
        ]
        
        # Log the command
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Record start time
        start_time = time.time()
        
        # Run the experiment
        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
            success = True
            error_msg = ""
        except subprocess.CalledProcessError as e:
            success = False
            error_msg = f"Command failed with code {e.returncode}: {e.stderr.decode('utf-8')}"
            logger.error(error_msg)
        
        # Record end time
        end_time = time.time()
        duration = end_time - start_time
        
        # Get experiment directory using the descriptive ID (not latest)
        target_dir = os.path.join(self.base_dir, f"experiment_{exp_id}")
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy the experiment data
        tmp_exp_dir = f"tmp/{desc_exp_id}"
        if os.path.exists(tmp_exp_dir):
            try:
                subprocess.run(f"cp -r {tmp_exp_dir}/* {target_dir}", shell=True, check=True)
                logger.info(f"Copied experiment data from {tmp_exp_dir} to {target_dir}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to copy experiment data: {e}")
        else:
            logger.error(f"Experiment directory '{tmp_exp_dir}' not found")

        # Remove tmp dir
        # logger.info("Removing tmp dir...")
        # os.remove(tmp_exp_dir)
        
        # Record experiment result
        result_record = {
            **params,
            'exp_id': desc_exp_id,
            'success': success,
            'runtime': duration,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'error': error_msg
        }
        
        self.experiment_summary.append(result_record)
        return result_record
    
    def run_all_experiments(self):
        """Run all experiments and collect results"""
        logger.info(f"Running {len(self.param_combinations)} experiments")
        
        for i, params in enumerate(self.param_combinations):
            logger.info(f"Running experiment {i+1}/{len(self.param_combinations)}")
            result = self.run_experiment(params)
            
            # Save intermediate summary after each experiment
            self._save_summary()
            
            # Wait between experiments to let the system settle
            time.sleep(5)
            
        logger.info("All experiments completed")
        return self.experiment_summary
    
    def _save_summary(self):
        """Save experiment summary to CSV"""
        df = pd.DataFrame(self.experiment_summary)
        df.to_csv(self.summary_file, index=False)
        logger.info(f"Summary saved to {self.summary_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Run multiple network experiments')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Output directory for experiment data')
    parser.add_argument('--config', type=str, default='',
                        help='Optional config file with experiment parameters')
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.output_dir, args.config)
    results = runner.run_all_experiments()
    
    # Print final summary
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"Experiments completed: {success_count}/{len(results)} succeeded")

if __name__ == "__main__":
    main()