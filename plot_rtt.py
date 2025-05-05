#!/usr/bin/env python3
"""
Generate RTT histograms from TCP flow RTT data using matplotlib.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_rtt_histogram(csv_file, output_dir):
    """Generate RTT histogram plot from CSV data."""
    try:
        # Read the RTT data
        logging.info(f"Reading RTT data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Check if we have RTT data
        if 'rtt_ms' not in df.columns:
            logging.error("No 'rtt_ms' column found in the CSV file")
            return False
            
        # Filter out None/NaN values
        df_filtered = df[df['rtt_ms'].notna()]
        
        if len(df_filtered) == 0:
            logging.error("No valid RTT measurements found in the data")
            return False
            
        logging.info(f"Found {len(df_filtered)} valid RTT measurements")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate histogram plot
        plt.figure(figsize=(10, 6))
        plt.hist(df_filtered['rtt_ms'], bins=50, alpha=0.75, color='blue', edgecolor='black')
        plt.title('RTT Distribution')
        plt.xlabel('RTT (ms)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # Add statistics as text
        rtt_mean = df_filtered['rtt_ms'].mean()
        rtt_median = df_filtered['rtt_ms'].median()
        rtt_min = df_filtered['rtt_ms'].min()
        rtt_max = df_filtered['rtt_ms'].max()
        rtt_p95 = df_filtered['rtt_ms'].quantile(0.95)
        
        stats_text = (
            f"Mean: {rtt_mean:.2f} ms\n"
            f"Median: {rtt_median:.2f} ms\n"
            f"Min: {rtt_min:.2f} ms\n"
            f"Max: {rtt_max:.2f} ms\n"
            f"95th %ile: {rtt_p95:.2f} ms"
        )
        
        plt.figtext(0.75, 0.7, stats_text, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the plot
        output_file = os.path.join(output_dir, 'rtt_histogram.png')
        plt.savefig(output_file, dpi=100)
        plt.close()
        
        logging.info(f"RTT histogram saved to {output_file}")
        
        # Create a boxplot for better outlier visualization
        plt.figure(figsize=(10, 4))
        plt.boxplot(df_filtered['rtt_ms'], vert=False)
        plt.title('RTT Boxplot')
        plt.xlabel('RTT (ms)')
        plt.grid(True, alpha=0.3)
        
        output_file_boxplot = os.path.join(output_dir, 'rtt_boxplot.png')
        plt.savefig(output_file_boxplot, dpi=100)
        plt.close()
        
        logging.info(f"RTT boxplot saved to {output_file_boxplot}")
        
        # Create a CDF plot
        plt.figure(figsize=(10, 6))
        sorted_data = np.sort(df_filtered['rtt_ms'])
        yvals = np.arange(1, len(sorted_data)+1)/float(len(sorted_data))
        plt.plot(sorted_data, yvals)
        plt.title('RTT Cumulative Distribution Function (CDF)')
        plt.xlabel('RTT (ms)')
        plt.ylabel('Cumulative Probability')
        plt.grid(True, alpha=0.3)
        
        # Add vertical lines for key percentiles
        for p, label in [(0.5, 'Median'), (0.95, '95th %ile'), (0.99, '99th %ile')]:
            percentile_val = df_filtered['rtt_ms'].quantile(p)
            plt.axvline(x=percentile_val, linestyle='--', color='red', alpha=0.5)
            plt.text(percentile_val, p, f' {label}', verticalalignment='center')
        
        output_file_cdf = os.path.join(output_dir, 'rtt_cdf.png')
        plt.savefig(output_file_cdf, dpi=100)
        plt.close()
        
        logging.info(f"RTT CDF plot saved to {output_file_cdf}")
        
        return True
    
    except Exception as e:
        logging.error(f"Error generating RTT plots: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate RTT histogram plots from CSV data')
    parser.add_argument('csv_file', help='Path to the CSV file with RTT data')
    parser.add_argument('-o', '--output-dir', help='Directory to save plots (defaults to same directory as CSV)')
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.dirname(args.csv_file) or '.'
    
    # Generate the plots
    success = plot_rtt_histogram(args.csv_file, output_dir)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 