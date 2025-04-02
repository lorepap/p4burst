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
import shutil
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
        bandwidth_values = [10, 50, 100]  # Mbps
        delay_values = [0.001]  # ms - fixed to small value
        
        # Define burst degree configurations
        burst_degree_configs = {
            'small': {
                'burst_servers': 8,
                'bursty_reply_size': 2000,
                'burst_interval': 0.02  # slower burst interval
            },
            'medium': {
                'burst_servers': 16,
                'bursty_reply_size': 5000,
                'burst_interval': 0.01  # medium burst interval
            },
            'large': {
                'burst_servers': 32,
                'bursty_reply_size': 10000,
                'burst_interval': 0.005  # faster burst interval
            }
        }
        
        # Define background load configurations
        bg_load_configs = {
            'small': {
                'flow_size': 1000,
                'bg_flow_iat': 0.2  # slower background flow inter-arrival time
            },
            'medium': {
                'flow_size': 5000,
                'bg_flow_iat': 0.1  # medium background flow inter-arrival time
            },
            'large': {
                'flow_size': 10000,
                'bg_flow_iat': 0.05  # faster background flow inter-arrival time
            }
        }
        
        # Generate parameter combinations
        combinations = []
        
        # Fixed parameters
        n_hosts = 32
        n_clients = 16
        n_servers = n_hosts - n_clients  # Calculate servers as remaining hosts
        
        experiment_id = 0
        for bw, delay, burst_degree, bg_load in itertools.product(
            bandwidth_values, 
            delay_values, 
            burst_degree_configs.keys(),
            bg_load_configs.keys()
        ):
            # Get burst and background configurations
            burst_config = burst_degree_configs[burst_degree]
            bg_config = bg_load_configs[bg_load]
            
            # Calculate experiment duration - longer for higher load scenarios
            duration = 5
            
            # Create parameter set
            params = {
                'experiment_id': experiment_id,
                'bw': bw,
                'delay': delay,
                'burst_degree': burst_degree,
                'bg_load': bg_load,
                'duration': duration,
                'n_hosts': n_hosts,  # Fixed parameters
                'n_leaf': 8,
                'n_spine': 4,
                'bursty_reply_size': burst_config['bursty_reply_size'],
                'n_clients': n_clients,
                'n_servers': n_servers,  # Calculated from hosts and clients
                'burst_servers': burst_config['burst_servers'],
                'burst_clients': 4,  # Fixed number of burst clients
                'flow_size': bg_config['flow_size'],
                'bg_flow_iat': bg_config['bg_flow_iat'],
                'burst_interval': burst_config['burst_interval'],
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
        desc_exp_id = f"bw_{params['bw']}_delay_{params['delay']}_burst_{params['burst_degree']}_bg_{params['bg_load']}"
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
            "--bg_flow_iat", str(params['bg_flow_iat']),
            "--flow_size", str(params['flow_size']),
            "--bursty_reply_size", str(params['bursty_reply_size']),
            "--burst_interval", str(params['burst_interval']),
            "--burst_servers", str(params['burst_servers']),
            "--burst_clients", str(params['burst_clients']),
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
                
                # Remove the temporary directory after successful copy
                shutil.rmtree(tmp_exp_dir)
                logger.info(f"Removed temporary directory {tmp_exp_dir}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to copy experiment data: {e}")
            except Exception as e:
                logger.error(f"Failed to remove temporary directory: {e}")
        else:
            logger.error(f"Experiment directory '{tmp_exp_dir}' not found")
        
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