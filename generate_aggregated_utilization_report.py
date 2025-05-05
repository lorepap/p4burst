#!/usr/bin/env python3

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

def generate_aggregated_utilization_report(output_dir):
    """
    Generate aggregated link utilization report across all clients.
    
    Args:
        output_dir: The base output directory where utilization files are stored
            and where to save the visualization report
    """
    util_dir = os.path.join(output_dir, "utilization")
    
    if not os.path.exists(util_dir):
        print(f"Utilization directory {util_dir} does not exist.")
        return False
    
    csv_files = glob.glob(os.path.join(util_dir, "*_utilization.csv"))
    if not csv_files:
        print("No utilization files found. Skipping aggregated report.")
        return False
    
    # Read all utilization data
    all_util_data = []
    for file in csv_files:
        client_name = os.path.basename(file).replace("_utilization.csv", "")
        try:
            df = pd.read_csv(file)
            df['client'] = client_name
            all_util_data.append(df)
        except Exception as e:
            print(f"Error reading utilization file {file}: {e}")
    
    if not all_util_data:
        print("Failed to read any utilization data. Skipping aggregated report.")
        return False
    
    combined_df = pd.concat(all_util_data)
    
    # Calculate per-client statistics
    client_stats = combined_df.groupby('client').agg({
        'utilization_percent': ['mean', 'max', 'min'],
        'throughput_mbps': ['mean', 'max', 'min']
    }).reset_index()
    
    # Reshape for better readability
    client_stats.columns = ['_'.join(col).strip('_') for col in client_stats.columns.values]
    
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create bar plot of average utilization per client
    plt.figure(figsize=(10, 6))
    bars = plt.bar(client_stats['client'], client_stats['utilization_percent_mean'])
    plt.title('Average Link Utilization by Client')
    plt.xlabel('Client')
    plt.ylabel('Utilization (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.savefig(os.path.join(plots_dir, 'avg_utilization_by_client.png'))
    plt.close()
    
    # Create bar plot of average throughput per client
    plt.figure(figsize=(10, 6))
    bars = plt.bar(client_stats['client'], client_stats['throughput_mbps_mean'])
    plt.title('Average Throughput by Client')
    plt.xlabel('Client')
    plt.ylabel('Throughput (Mbps)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f} Mbps', ha='center', va='bottom')
    
    plt.savefig(os.path.join(plots_dir, 'avg_throughput_by_client.png'))
    plt.close()
    
    # Create time series plots for each client
    for client in combined_df['client'].unique():
        client_df = combined_df[combined_df['client'] == client]
        
        plt.figure(figsize=(12, 6))
        plt.plot(client_df['time_sec'], client_df['throughput_mbps'])
        plt.title(f'Throughput Over Time - {client}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Throughput (Mbps)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'{client}_throughput_time_series.png'))
        plt.close()
    
    # Create overall time series with all clients for throughput
    plt.figure(figsize=(14, 8))
    for client in combined_df['client'].unique():
        client_df = combined_df[combined_df['client'] == client]
        plt.plot(client_df['time_sec'], client_df['throughput_mbps'], label=client)
    
    plt.title('Throughput Over Time - All Clients')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Throughput (Mbps)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_clients_throughput_time_series.png'))
    plt.close()
    
    # Create overall time series with all clients for utilization
    plt.figure(figsize=(14, 8))
    for client in combined_df['client'].unique():
        client_df = combined_df[combined_df['client'] == client]
        plt.plot(client_df['time_sec'], client_df['utilization_percent'], label=client)
    
    plt.title('Link Utilization Over Time - All Clients')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Utilization (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'all_clients_utilization_time_series.png'))
    plt.close()
    
    # Write summary report
    report_file = os.path.join(output_dir, "link_utilization_report.txt")
    with open(report_file, 'w') as f:
        f.write("=== Link Utilization Report ===\n\n")
        f.write("== Per-Client Statistics ==\n\n")
        
        for _, row in client_stats.iterrows():
            f.write(f"Client: {row['client']}\n")
            f.write(f"  Average Utilization: {row['utilization_percent_mean']:.2f}%\n")
            f.write(f"  Peak Utilization: {row['utilization_percent_max']:.2f}%\n")
            f.write(f"  Average Throughput: {row['throughput_mbps_mean']:.2f} Mbps\n")
            f.write(f"  Peak Throughput: {row['throughput_mbps_max']:.2f} Mbps\n\n")
        
        # Overall statistics
        f.write("== Overall Statistics ==\n\n")
        f.write(f"Average Utilization Across All Clients: {combined_df['utilization_percent'].mean():.2f}%\n")
        f.write(f"Peak Utilization Across All Clients: {combined_df['utilization_percent'].max():.2f}%\n")
        f.write(f"Average Throughput Across All Clients: {combined_df['throughput_mbps'].mean():.2f} Mbps\n")
        f.write(f"Peak Throughput Across All Clients: {combined_df['throughput_mbps'].max():.2f} Mbps\n")
    
    print(f"Link utilization report generated at {report_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate aggregated link utilization report')
    parser.add_argument('output_dir', help='Directory containing utilization data and where to save the report')
    args = parser.parse_args()
    
    generate_aggregated_utilization_report(args.output_dir)

if __name__ == '__main__':
    main() 