#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_qct_stats(stats_dir, output_dir=None):
    """
    Combine QCT statistics from multiple files into one dataframe and save to CSV.
    
    Args:
        stats_dir: Directory containing QCT stats CSV files
        output_dir: Directory to save combined stats and plots (defaults to stats_dir)
    
    Returns:
        Combined dataframe
    """
    if output_dir is None:
        output_dir = stats_dir
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Find all QCT stats files
    qct_files = glob.glob(os.path.join(stats_dir, "*_stats.csv"))
    flow_files = glob.glob(os.path.join(stats_dir, "*_flows.csv"))
    
    if not qct_files:
        logging.error(f"No QCT stats files found in {stats_dir}")
        return None, None
    
    # Combine QCT stats
    all_qct = []
    for file in qct_files:
        try:
            client_name = os.path.basename(file).replace("_stats.csv", "")
            df = pd.read_csv(file)
            df['client'] = client_name
            all_qct.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    
    if not all_qct:
        logging.error("Failed to read any QCT stats files")
        return None, None
    
    combined_qct = pd.concat(all_qct, ignore_index=True)
    
    # Combine flow stats if available
    all_flows = []
    for file in flow_files:
        try:
            client_name = os.path.basename(file).replace("_flows.csv", "")
            df = pd.read_csv(file)
            df['client'] = client_name
            all_flows.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
    
    combined_flows = None
    if all_flows:
        combined_flows = pd.concat(all_flows, ignore_index=True)
    
    # Save combined stats
    combined_qct_path = os.path.join(output_dir, "combined_qct_stats.csv")
    combined_qct.to_csv(combined_qct_path, index=False)
    logging.info(f"Saved combined QCT stats to {combined_qct_path}")
    
    if combined_flows is not None:
        combined_flows_path = os.path.join(output_dir, "combined_flow_stats.csv")
        combined_flows.to_csv(combined_flows_path, index=False)
        logging.info(f"Saved combined flow stats to {combined_flows_path}")
    
    # Generate QCT visualizations
    generate_qct_visualizations(combined_qct, combined_flows, plots_dir)
    
    return combined_qct, combined_flows

def generate_qct_visualizations(qct_df, flow_df, plots_dir):
    """
    Generate visualizations for QCT data.
    
    Args:
        qct_df: DataFrame containing QCT statistics
        flow_df: DataFrame containing per-flow statistics (can be None)
        plots_dir: Directory to save plots
    """
    if qct_df is None or len(qct_df) == 0:
        logging.error("No QCT data to visualize")
        return
    
    try:
        # Filter out entries with no QCT value
        valid_qct_df = qct_df[qct_df['qct'].notna() & (qct_df['qct'] != '')].copy()
        
        # If all entries are invalid, handle this edge case
        if len(valid_qct_df) == 0:
            logging.error("No valid QCT data available")
            return
            
        # Convert time strings to numeric if needed
        if isinstance(valid_qct_df['qct'].iloc[0], str):
            valid_qct_df['qct'] = pd.to_numeric(valid_qct_df['qct'])
        
        # Convert to milliseconds for better readability
        valid_qct_df['qct_ms'] = valid_qct_df['qct'] * 1000
        
        # 1. QCT Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(valid_qct_df['qct_ms'], bins=30, alpha=0.7, color='blue')
        plt.title('Query Completion Time (QCT) Distribution')
        plt.xlabel('QCT (ms)')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "qct_histogram.png"))
        plt.close()
        
        # 2. QCT CDF
        plt.figure(figsize=(10, 6))
        sorted_qct = np.sort(valid_qct_df['qct_ms'])
        yvals = np.arange(1, len(sorted_qct)+1) / len(sorted_qct)
        plt.plot(sorted_qct, yvals, marker='.', linestyle='none')
        plt.title('Query Completion Time (QCT) Cumulative Distribution')
        plt.xlabel('QCT (ms)')
        plt.ylabel('CDF')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "qct_cdf.png"))
        plt.close()
        
        # 3. QCT Boxplot (by client if multiple clients)
        plt.figure(figsize=(10, 6))
        if 'client' in valid_qct_df.columns and len(valid_qct_df['client'].unique()) > 1:
            valid_qct_df.boxplot(column='qct_ms', by='client', grid=False)
            plt.title('Query Completion Time (QCT) by Client')
        else:
            plt.boxplot(valid_qct_df['qct_ms'])
            plt.title('Query Completion Time (QCT)')
        plt.ylabel('QCT (ms)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plots_dir, "qct_boxplot.png"))
        plt.close()
        
        # 4. QCT by number of servers (if available)
        if 'num_servers' in valid_qct_df.columns:
            plt.figure(figsize=(10, 6))
            server_groups = valid_qct_df.groupby('num_servers')['qct_ms'].agg(['mean', 'median', 'std']).reset_index()
            
            plt.errorbar(server_groups['num_servers'], server_groups['mean'], 
                         yerr=server_groups['std'], fmt='o-', capsize=5)
            plt.title('QCT vs. Number of Servers in Query')
            plt.xlabel('Number of Servers')
            plt.ylabel('Average QCT (ms)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "qct_by_servers.png"))
            plt.close()
        
        # 5. QCT vs Response Size (if available)
        if 'bytes_received' in valid_qct_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_qct_df['bytes_received'] / 1000, valid_qct_df['qct_ms'], alpha=0.7)
            plt.title('QCT vs. Response Size')
            plt.xlabel('Response Size (KB)')
            plt.ylabel('QCT (ms)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "qct_vs_size.png"))
            plt.close()
        
        # Analyze flow-level data if available
        if flow_df is not None and len(flow_df) > 0:
            # Filter out entries with no FCT value
            valid_flow_df = flow_df[flow_df['fct'].notna() & (flow_df['fct'] != '')].copy()
                
            if len(valid_flow_df) == 0:
                logging.error("No valid flow data available")
                return
                
            if 'fct' in valid_flow_df.columns:
                # Convert time strings to numeric if needed
                if isinstance(valid_flow_df['fct'].iloc[0], str):
                    valid_flow_df['fct'] = pd.to_numeric(valid_flow_df['fct'])
                
                # Convert to milliseconds for better readability
                valid_flow_df['fct_ms'] = valid_flow_df['fct'] * 1000
                
                # 6. Flow Completion Time Histogram
                plt.figure(figsize=(10, 6))
                plt.hist(valid_flow_df['fct_ms'], bins=30, alpha=0.7, color='green')
                plt.title('Flow Completion Time (FCT) Distribution Within Queries')
                plt.xlabel('FCT (ms)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(plots_dir, "query_flow_fct_histogram.png"))
                plt.close()
                
                # 7. QCT vs max flow FCT (check relationship between QCT and slowest flow)
                if 'query_id' in valid_flow_df.columns:
                    max_fct_by_query = valid_flow_df.groupby('query_id')['fct_ms'].max().reset_index()
                    max_fct_by_query.columns = ['query_id', 'max_flow_fct_ms']
                    
                    if 'query_id' in valid_qct_df.columns:
                        merged_df = pd.merge(valid_qct_df, max_fct_by_query, on='query_id', how='inner')
                        
                        if len(merged_df) > 0:
                            plt.figure(figsize=(10, 6))
                            plt.scatter(merged_df['max_flow_fct_ms'], merged_df['qct_ms'], alpha=0.7)
                            
                            # Add y=x line to show QCT vs max FCT
                            max_val = max(merged_df['max_flow_fct_ms'].max(), merged_df['qct_ms'].max())
                            plt.plot([0, max_val], [0, max_val], 'r--', label='y=x')
                            
                            plt.title('QCT vs. Max Flow FCT')
                            plt.xlabel('Max Flow FCT in Query (ms)')
                            plt.ylabel('Query Completion Time (ms)')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.savefig(os.path.join(plots_dir, "qct_vs_max_fct.png"))
                            plt.close()
        
        logging.info(f"Generated QCT visualizations in {plots_dir}")
        
    except Exception as e:
        logging.error(f"Error generating QCT visualizations: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Combine QCT statistics from multiple bursty clients')
    parser.add_argument('stats_dir', help='Directory containing QCT stats CSV files')
    parser.add_argument('-o', '--output-dir', help='Directory to save combined stats and plots')
    
    args = parser.parse_args()
    
    stats_dir = args.stats_dir
    output_dir = args.output_dir if args.output_dir else stats_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine stats and generate visualizations
    combine_qct_stats(stats_dir, output_dir)
    
    return 0

if __name__ == "__main__":
    main() 