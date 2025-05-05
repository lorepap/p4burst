#!/usr/bin/env python3
"""
Combine FCT statistics from multiple clients and generate visualizations.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_client_ip(filename):
    """Extract client IP from the filename."""
    match = re.search(r'bg_client_(\d+\.\d+\.\d+\.\d+)', os.path.basename(filename))
    if match:
        return match.group(1)
    return os.path.basename(filename)  # Fallback to filename if pattern not found

def combine_fct_stats(input_dir, output_file):
    """Combine all FCT statistics from individual clients into a single CSV file."""
    stats_files = glob.glob(os.path.join(input_dir, '*_stats.csv'))
    
    if not stats_files:
        logging.error(f"No FCT statistics files found in {input_dir}")
        return None
    
    # Filter to include only bg_client files for FCT
    client_stats_files = [f for f in stats_files if 'bg_client' in f]
    
    if not client_stats_files:
        logging.error(f"No client FCT statistics files found in {input_dir}")
        return None
    
    logging.info(f"Found {len(client_stats_files)} client FCT statistics files")
    
    all_stats = []
    for file_path in client_stats_files:
        try:
            df = pd.read_csv(file_path)
            # Add source file information to identify the origin
            df['source_file'] = os.path.basename(file_path)
            # Add client IP for better identification
            df['client_ip'] = extract_client_ip(file_path)
            all_stats.append(df)
            logging.info(f"Loaded {len(df)} rows from {file_path}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
    
    if not all_stats:
        logging.error("No valid FCT statistics found")
        return None
    
    combined_df = pd.concat(all_stats, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    logging.info(f"Combined FCT statistics saved to {output_file} with {len(combined_df)} rows")
    
    return combined_df

def combine_rtt_stats(input_dir, output_file):
    """Combine all packet RTT statistics from individual clients into a single CSV file."""
    rtt_files = glob.glob(os.path.join(input_dir, '*_rtt.csv'))
    
    if not rtt_files:
        logging.error(f"No RTT statistics files found in {input_dir}")
        return None
    
    logging.info(f"Found {len(rtt_files)} RTT statistics files")
    
    all_rtts = []
    for file_path in rtt_files:
        try:
            df = pd.read_csv(file_path)
            # Filter out missing RTT values right at load time
            df = df[df['rtt_ms'].notna()]
            if len(df) == 0:
                logging.warning(f"No valid RTT measurements in {file_path}")
                continue
                
            # Add source file information to identify the origin
            df['source_file'] = os.path.basename(file_path)
            # Add client/server IP for better identification
            df['endpoint_ip'] = extract_client_ip(file_path)
            # Add endpoint type (client or server)
            df['endpoint_type'] = 'client' if 'bg_client' in os.path.basename(file_path) else 'server'
            all_rtts.append(df)
            logging.info(f"Loaded {len(df)} rows with valid RTT from {file_path}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
    
    if not all_rtts:
        logging.error("No valid RTT statistics found")
        return None
    
    combined_df = pd.concat(all_rtts, ignore_index=True)
    
    # Double-check for any remaining NaN values
    combined_df = combined_df[combined_df['rtt_ms'].notna()]
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    logging.info(f"Combined RTT statistics saved to {output_file} with {len(combined_df)} rows")
    
    return combined_df

def plot_rtt_visualizations(df, output_dir):
    """Generate visualizations for the combined RTT data."""
    if 'rtt_ms' not in df.columns:
        logging.error("No 'rtt_ms' column found in the combined RTT data")
        return False
    
    # Filter out None/NaN values (should already be filtered, but double-check)
    df = df[df['rtt_ms'].notna()]
    
    if len(df) == 0:
        logging.error("No valid RTT measurements found in the data")
        return False
    
    # Create plots directory within the output directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logging.info(f"Saving RTT visualizations to {plots_dir}")
    
    # Get RTT values in milliseconds
    rtt_ms = df['rtt_ms']
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(rtt_ms, bins=50, alpha=0.75, color='green', edgecolor='black')
    plt.title('RTT Distribution')
    plt.xlabel('RTT (ms)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    rtt_mean = rtt_ms.mean()
    rtt_median = rtt_ms.median()
    rtt_min = rtt_ms.min()
    rtt_max = rtt_ms.max()
    rtt_p95 = np.percentile(rtt_ms, 95)
    rtt_p99 = np.percentile(rtt_ms, 99)
    
    stats_text = (
        f"Mean: {rtt_mean:.2f} ms\n"
        f"Median: {rtt_median:.2f} ms\n"
        f"Min: {rtt_min:.2f} ms\n"
        f"Max: {rtt_max:.2f} ms\n"
        f"95th %ile: {rtt_p95:.2f} ms\n"
        f"99th %ile: {rtt_p99:.2f} ms"
    )
    
    plt.figtext(0.75, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the histogram
    output_file = os.path.join(plots_dir, 'rtt_histogram.png')
    plt.savefig(output_file, dpi=100)
    plt.close()
    
    logging.info(f"RTT histogram saved to {output_file}")
    
    # Create boxplot
    plt.figure(figsize=(10, 4))
    plt.boxplot(rtt_ms, vert=False)
    plt.title('RTT Boxplot')
    plt.xlabel('RTT (ms)')
    plt.grid(True, alpha=0.3)
    
    # Save the boxplot
    output_file_boxplot = os.path.join(plots_dir, 'rtt_boxplot.png')
    plt.savefig(output_file_boxplot, dpi=100)
    plt.close()
    
    logging.info(f"RTT boxplot saved to {output_file_boxplot}")
    
    # Create CDF plot
    plt.figure(figsize=(10, 6))
    sorted_data = np.sort(rtt_ms)
    yvals = np.arange(1, len(sorted_data)+1)/float(len(sorted_data))
    plt.plot(sorted_data, yvals)
    plt.title('RTT Cumulative Distribution Function (CDF)')
    plt.xlabel('RTT (ms)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for key percentiles
    for p, label in [(0.5, 'Median'), (0.95, '95th %ile'), (0.99, '99th %ile')]:
        percentile_val = np.percentile(rtt_ms, p*100)
        plt.axvline(x=percentile_val, linestyle='--', color='red', alpha=0.5)
        plt.text(percentile_val, p, f' {label}', verticalalignment='center')
    
    # Save the CDF plot
    output_file_cdf = os.path.join(plots_dir, 'rtt_cdf.png')
    plt.savefig(output_file_cdf, dpi=100)
    plt.close()
    
    logging.info(f"RTT CDF plot saved to {output_file_cdf}")
    
    # If there are multiple flows, create RTT over time analysis
    if 'send_time' in df.columns and 'ack_time' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.scatter(df['send_time'], rtt_ms, alpha=0.5, s=5)
        plt.title('RTT Over Time')
        plt.xlabel('Send Time (s)')
        plt.ylabel('RTT (ms)')
        plt.grid(True, alpha=0.3)
        
        # Save the time series plot
        output_file_time = os.path.join(plots_dir, 'rtt_over_time.png')
        plt.savefig(output_file_time, dpi=100)
        plt.close()
        
        logging.info(f"RTT over time plot saved to {output_file_time}")
        
        # Create a client-only version of the RTT over time plot if endpoint_type is available
        if 'endpoint_type' in df.columns:
            client_df = df[df['endpoint_type'] == 'client']
            if not client_df.empty:
                plt.figure(figsize=(12, 6))
                plt.scatter(client_df['send_time'], client_df['rtt_ms'], alpha=0.5, s=5)
                plt.title('Client RTT Over Time')
                plt.xlabel('Send Time (s)')
                plt.ylabel('RTT (ms)')
                plt.grid(True, alpha=0.3)
                
                # Save the client-only time series plot
                output_file_client_time = os.path.join(plots_dir, 'client_rtt_over_time.png')
                plt.savefig(output_file_client_time, dpi=100)
                plt.close()
                
                logging.info(f"Client RTT over time plot saved to {output_file_client_time}")
    
    # If we have endpoint IPs, generate a comparison
    if 'endpoint_ip' in df.columns and 'source_file' in df.columns:
        # Filter out server endpoints for the endpoint comparison plot
        client_df = df[~df['source_file'].str.contains('bg_server')]
        
        # Get RTT statistics per endpoint (clients only)
        endpoints = client_df['endpoint_ip'].unique()
        
        if len(endpoints) > 1:
            # Calculate statistics by endpoint
            endpoint_stats = client_df.groupby('endpoint_ip')['rtt_ms'].agg(['mean', 'median', 'min', 'max', 'count'])
            endpoint_stats = endpoint_stats.sort_values('mean', ascending=False)
            
            # Bar chart of average RTT per endpoint
            plt.figure(figsize=(12, 6))
            endpoint_stats['mean'].plot(kind='bar', yerr=endpoint_stats['max']-endpoint_stats['mean'])
            plt.title('Average RTT by Client Endpoint')
            plt.ylabel('RTT (ms)')
            plt.xlabel('Client Endpoint')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save endpoint comparison chart
            output_file_endpoints = os.path.join(plots_dir, 'rtt_by_endpoint.png')
            plt.savefig(output_file_endpoints, dpi=100)
            plt.close()
            
            logging.info(f"RTT by client endpoint chart saved to {output_file_endpoints}")
    
    # If there are multiple clients, generate a comparison across clients
    if 'flow_id' in df.columns:
        # Get RTT statistics per flow
        flows = df['flow_id'].unique()
        
        if len(flows) > 1:
            # Limit to top 10 flows with most packets
            flow_counts = df['flow_id'].value_counts().head(10)
            top_flows = flow_counts.index.tolist()
            
            # Filter for top flows
            df_top = df[df['flow_id'].isin(top_flows)]
            
            # Create boxplot by flow
            plt.figure(figsize=(14, 8))
            plt.boxplot([df_top[df_top['flow_id'] == flow]['rtt_ms'] for flow in top_flows],
                       labels=[f"Flow {i+1}" for i in range(len(top_flows))],
                       vert=False)
            plt.title('RTT by Flow (Top 10 Flows)')
            plt.xlabel('RTT (ms)')
            plt.grid(True, alpha=0.3)
            
            # Save the flow comparison boxplot
            output_file_flows = os.path.join(plots_dir, 'rtt_by_flow.png')
            plt.savefig(output_file_flows, dpi=100)
            plt.close()
            
            logging.info(f"RTT by flow boxplot saved to {output_file_flows}")
    
    return True

def plot_fct_visualizations(df, output_dir):
    """Generate visualizations for the combined FCT data."""
    if 'fct' not in df.columns:
        logging.error("No 'fct' column found in the combined data")
        return False
    
    # Create plots directory within the output directory
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    logging.info(f"Saving visualizations to {plots_dir}")
    
    # Convert FCT from seconds to milliseconds for better readability
    fct_ms = df['fct'] * 1000
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(fct_ms, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title('FCT Distribution')
    plt.xlabel('FCT (ms)')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Add statistics as text
    fct_mean = fct_ms.mean()
    fct_median = fct_ms.median()
    fct_min = fct_ms.min()
    fct_max = fct_ms.max()
    fct_p95 = np.percentile(fct_ms, 95)
    fct_p99 = np.percentile(fct_ms, 99)
    
    stats_text = (
        f"Mean: {fct_mean:.2f} ms\n"
        f"Median: {fct_median:.2f} ms\n"
        f"Min: {fct_min:.2f} ms\n"
        f"Max: {fct_max:.2f} ms\n"
        f"95th %ile: {fct_p95:.2f} ms\n"
        f"99th %ile: {fct_p99:.2f} ms"
    )
    
    plt.figtext(0.75, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the histogram
    output_file = os.path.join(plots_dir, 'fct_histogram.png')
    plt.savefig(output_file, dpi=100)
    plt.close()
    
    logging.info(f"FCT histogram saved to {output_file}")
    
    # Create boxplot
    plt.figure(figsize=(10, 4))
    plt.boxplot(fct_ms, vert=False)
    plt.title('FCT Boxplot')
    plt.xlabel('FCT (ms)')
    plt.grid(True, alpha=0.3)
    
    # Save the boxplot
    output_file_boxplot = os.path.join(plots_dir, 'fct_boxplot.png')
    plt.savefig(output_file_boxplot, dpi=100)
    plt.close()
    
    logging.info(f"FCT boxplot saved to {output_file_boxplot}")
    
    # Create CDF plot
    plt.figure(figsize=(10, 6))
    sorted_data = np.sort(fct_ms)
    yvals = np.arange(1, len(sorted_data)+1)/float(len(sorted_data))
    plt.plot(sorted_data, yvals)
    plt.title('FCT Cumulative Distribution Function (CDF)')
    plt.xlabel('FCT (ms)')
    plt.ylabel('Cumulative Probability')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for key percentiles
    for p, label in [(0.5, 'Median'), (0.95, '95th %ile'), (0.99, '99th %ile')]:
        percentile_val = np.percentile(fct_ms, p*100)
        plt.axvline(x=percentile_val, linestyle='--', color='red', alpha=0.5)
        plt.text(percentile_val, p, f' {label}', verticalalignment='center')
    
    # Save the CDF plot
    output_file_cdf = os.path.join(plots_dir, 'fct_cdf.png')
    plt.savefig(output_file_cdf, dpi=100)
    plt.close()
    
    logging.info(f"FCT CDF plot saved to {output_file_cdf}")
    
    # If there are multiple clients, generate a comparison across clients
    if 'client_ip' in df.columns:
        # Get FCT statistics per client using IP
        clients = df['client_ip'].unique()
        
        if len(clients) > 1:
            client_stats = df.groupby('client_ip')['fct'].agg(['mean', 'median', 'min', 'max'])
            client_stats = client_stats.sort_values('mean', ascending=False)
            
            # Bar chart of average FCT per client
            plt.figure(figsize=(12, 6))
            # Convert to ms for better readability
            client_stats['mean'] *= 1000
            client_stats['median'] *= 1000
            client_stats['min'] *= 1000
            client_stats['max'] *= 1000
            
            client_stats['mean'].plot(kind='bar', yerr=client_stats['max']-client_stats['mean'])
            plt.title('Average FCT by Client')
            plt.ylabel('FCT (ms)')
            plt.xlabel('Client IP')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save client comparison chart
            output_file_clients = os.path.join(plots_dir, 'fct_by_client.png')
            plt.savefig(output_file_clients, dpi=100)
            plt.close()
            
            logging.info(f"FCT by client chart saved to {output_file_clients}")
    elif 'src' in df.columns or 'source_file' in df.columns:
        # For backward compatibility, use existing columns if client_ip not available
        group_col = 'src' if 'src' in df.columns else 'source_file'
        
        # Get FCT statistics per client
        clients = df[group_col].unique()
        
        if len(clients) > 1:
            client_stats = df.groupby(group_col)['fct'].agg(['mean', 'median', 'min', 'max'])
            client_stats = client_stats.sort_values('mean', ascending=False)
            
            # Bar chart of average FCT per client
            plt.figure(figsize=(12, 6))
            # Convert to ms for better readability
            client_stats['mean'] *= 1000
            client_stats['median'] *= 1000
            client_stats['min'] *= 1000
            client_stats['max'] *= 1000
            
            client_stats['mean'].plot(kind='bar', yerr=client_stats['max']-client_stats['mean'])
            plt.title('Average FCT by Client')
            plt.ylabel('FCT (ms)')
            plt.xlabel('Client')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Save client comparison chart
            output_file_clients = os.path.join(plots_dir, 'fct_by_client.png')
            plt.savefig(output_file_clients, dpi=100)
            plt.close()
            
            logging.info(f"FCT by client chart saved to {output_file_clients}")
    
    # Check if packet size or flow size is available for additional insights
    if 'size' in df.columns or 'flow_size' in df.columns or 'packet_size' in df.columns:
        # Identify the size column
        size_col = None
        for col in ['size', 'flow_size', 'packet_size']:
            if col in df.columns:
                size_col = col
                break
        
        if size_col:
            # Scatter plot of FCT vs size
            plt.figure(figsize=(10, 6))
            plt.scatter(df[size_col], fct_ms, alpha=0.5)
            plt.title(f'FCT vs {size_col.replace("_", " ").title()}')
            plt.xlabel(f'{size_col.replace("_", " ").title()}')
            plt.ylabel('FCT (ms)')
            plt.grid(True, alpha=0.3)
            
            # Save the scatter plot
            output_file_scatter = os.path.join(plots_dir, f'fct_vs_{size_col}.png')
            plt.savefig(output_file_scatter, dpi=100)
            plt.close()
            
            logging.info(f"FCT vs {size_col} scatter plot saved to {output_file_scatter}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Combine FCT statistics from multiple clients and generate visualizations')
    parser.add_argument('input_dir', help='Directory containing FCT statistics files')
    parser.add_argument('-o', '--output-dir', help='Directory to save combined stats and visualizations (defaults to input directory)')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    output_dir = args.output_dir or args.input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine FCT statistics
    output_file = os.path.join(output_dir, 'combined_fct_stats.csv')
    combined_df = combine_fct_stats(args.input_dir, output_file)
    
    if combined_df is not None:
        # Generate FCT visualizations
        success = plot_fct_visualizations(combined_df, output_dir)
        if success:
            logging.info("FCT visualizations generated successfully")
        else:
            logging.error("Failed to generate FCT visualizations")
    else:
        logging.error("Failed to combine FCT statistics")
    
    # Combine RTT statistics
    rtt_output_file = os.path.join(output_dir, 'combined_rtt_stats.csv')
    combined_rtt_df = combine_rtt_stats(args.input_dir, rtt_output_file)
    
    if combined_rtt_df is not None:
        # Generate RTT visualizations
        success = plot_rtt_visualizations(combined_rtt_df, output_dir)
        if success:
            logging.info("RTT visualizations generated successfully")
        else:
            logging.error("Failed to generate RTT visualizations")
    else:
        logging.error("Failed to combine RTT statistics")
    
    logging.info("Analysis complete")
    
    if (combined_df is None and combined_rtt_df is None):
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main()) 