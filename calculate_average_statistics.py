#!/usr/bin/env python3

import os
import glob
import pandas as pd
import numpy as np
import argparse

def calculate_average_statistics(stats_dir, summary_file):
    """
    Calculate average statistics across client flows only and append to summary file.
    
    Args:
        stats_dir: Directory containing the FCT and RTT statistics files
        summary_file: File to append the summary statistics to
    """
    # Filter to include only client stats files
    csv_files = [f for f in glob.glob(os.path.join(stats_dir, '*_stats.csv')) if 'bg_client' in f]
    rtt_files = [f for f in glob.glob(os.path.join(stats_dir, '*_stats_rtt.csv')) if 'bg_client' in f]

    print(f'Found {len(csv_files)} client FCT files and {len(rtt_files)} client RTT files')

    if csv_files:
        # Process FCT stats
        all_stats = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                if 'fct' in df.columns:
                    all_stats.append(df)
            except Exception as e:
                print(f'Error reading {f}: {e}')
        
        if all_stats:
            combined = pd.concat(all_stats)
            mean_fct = combined['fct'].mean() * 1000  # Convert to ms
            median_fct = combined['fct'].median() * 1000
            min_fct = combined['fct'].min() * 1000
            max_fct = combined['fct'].max() * 1000
            
            with open(summary_file, 'a') as f:
                f.write('\nAggregate Client FCT Statistics:\n')
                f.write(f'  Average FCT: {mean_fct:.2f} ms\n')
                f.write(f'  Median FCT: {median_fct:.2f} ms\n')
                f.write(f'  Min FCT: {min_fct:.2f} ms\n')
                f.write(f'  Max FCT: {max_fct:.2f} ms\n')
                f.write(f'  Total client flows: {len(combined)}\n')
    
    # Process RTT stats for clients only
    if rtt_files:
        all_rtts = []
        for f in rtt_files:
            try:
                df = pd.read_csv(f)
                if 'rtt_ms' in df.columns:
                    valid_rtts = df[df['rtt_ms'].notna()]
                    if not valid_rtts.empty:
                        all_rtts.append(valid_rtts)
            except Exception as e:
                print(f'Error reading {f}: {e}')
        
        if all_rtts:
            combined_rtts = pd.concat(all_rtts)
            mean_rtt = combined_rtts['rtt_ms'].mean()
            median_rtt = combined_rtts['rtt_ms'].median()
            min_rtt = combined_rtts['rtt_ms'].min()
            max_rtt = combined_rtts['rtt_ms'].max()
            p95_rtt = np.percentile(combined_rtts['rtt_ms'], 95)
            
            with open(summary_file, 'a') as f:
                f.write('\nAggregate Client RTT Statistics:\n')
                f.write(f'  Average RTT: {mean_rtt:.2f} ms\n')
                f.write(f'  Median RTT: {median_rtt:.2f} ms\n')
                f.write(f'  Min RTT: {min_rtt:.2f} ms\n')
                f.write(f'  Max RTT: {max_rtt:.2f} ms\n')
                f.write(f'  95th percentile: {p95_rtt:.2f} ms\n')
                f.write(f'  Total client packets with RTT: {len(combined_rtts)}\n')

def main():
    parser = argparse.ArgumentParser(description='Calculate average statistics across client flows')
    parser.add_argument('stats_dir', help='Directory containing the FCT and RTT statistics files')
    parser.add_argument('summary_file', help='File to append the summary statistics to')
    args = parser.parse_args()
    
    calculate_average_statistics(args.stats_dir, args.summary_file)

if __name__ == '__main__':
    main() 