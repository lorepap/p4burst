#!/usr/bin/env python3

import os
import glob
import csv
import argparse
import pandas as pd
import subprocess

def count_flows_from_pcap(pcap_file):
    """Count flows from a PCAP file using tshark."""
    print(f"Analyzing {pcap_file}...")
    
    # Get flow information using tshark
    cmd = [
        "tshark", "-r", pcap_file,
        "-Y", "tcp",  # Filter for TCP traffic
        "-T", "fields",
        "-e", "ip.src", "-e", "ip.dst", 
        "-e", "tcp.srcport", "-e", "tcp.dstport",
        "-e", "tcp.flags", "-e", "frame.time_epoch"
    ]
    
    try:
        # Run tshark command
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Process the output
        lines = result.stdout.strip().split('\n')
        
        # Skip empty results
        if not lines or lines[0] == '':
            print(f"No TCP traffic found in {pcap_file}")
            return {
                'total_flows': 0,
                'successful_flows': 0
            }
        
        # Parse each line to identify unique flows
        flows = {}
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # The output format is tab-separated
            parts = line.split('\t')
            if len(parts) < 5:
                continue
                
            src_ip, dst_ip, src_port, dst_port, tcp_flags = parts[:5]
            
            # Create flow identifier (consistent direction for client->server flow)
            if int(src_port) > int(dst_port):
                # Swap so we always have client (higher port) -> server (lower port)
                flow_id = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}"
                is_client_to_server = False
            else:
                flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}"
                is_client_to_server = True
            
            # Initialize flow record if not exists
            if flow_id not in flows:
                flows[flow_id] = {
                    'has_syn': False, 
                    'has_synack': False,
                    'has_fin': False,
                    'has_rst': False,
                    'has_data': False
                }
            
            # Check TCP flags - match the exact format we see in the output
            # SYN flag
            if tcp_flags == '0x00000002':
                flows[flow_id]['has_syn'] = True
            
            # SYN-ACK flag    
            elif tcp_flags == '0x00000012':
                flows[flow_id]['has_synack'] = True
            
            # FIN or FIN-ACK flag
            elif tcp_flags == '0x00000011' or tcp_flags == '0x00000001':
                flows[flow_id]['has_fin'] = True
            
            # RST flag
            elif tcp_flags == '0x00000004':
                flows[flow_id]['has_rst'] = True
            
            # ACK flag (indicates data packet)
            elif tcp_flags == '0x00000010':
                flows[flow_id]['has_data'] = True
            
            # PSH-ACK flag (also indicates data packet)
            elif tcp_flags == '0x00000018':
                flows[flow_id]['has_data'] = True
        
        # Count total flows (those with SYN flag)
        total_flows = sum(1 for f in flows.values() if f['has_syn'])
        
        # Count successfully completed flows (those with SYN, SYN-ACK, data and FIN)
        successful_flows = sum(1 for f in flows.values() 
                              if f['has_syn'] and f['has_synack'] and 
                                 f['has_data'] and f['has_fin'] and 
                                 not f['has_rst'])
        
        print(f"Detected {total_flows} total flows and {successful_flows} successful flows in {pcap_file}")
        
        return {
            'total_flows': total_flows,
            'successful_flows': successful_flows
        }
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing {pcap_file}: {e}")
        return {
            'total_flows': 0,
            'successful_flows': 0
        }
    
def analyze_experiment_pcaps(data_dir, output_dir):
    """Analyze all PCAP files for an experiment and generate a CSV with flow counts."""
    # Find all PCAP files for this experiment
    pcap_pattern = os.path.join(data_dir, "bg_client*.pcap")
    pcap_files = glob.glob(pcap_pattern)
    
    if not pcap_files:
        print(f"No PCAP files found for experiment")
        return
    
    print(f"Found {len(pcap_files)} PCAP files to analyze")
    
    # Process each PCAP file
    results = []
    for pcap_file in pcap_files:
        filename = os.path.basename(pcap_file)
        
        # Determine traffic type from filename
        traffic_type = "background"
        
            
        # Extract client IP from filename
        try:
            client_ip = filename.split('_')[2]
        except IndexError:
            client_ip = "unknown"


        # Count flows
        flow_counts = count_flows_from_pcap(pcap_file)
        
        # Add to results
        results.append({
            'pcap_file': filename,
            'traffic_type': traffic_type,
            'client_ip': client_ip,
            'total_flows_sent': flow_counts['total_flows'],
            'successful_flows': flow_counts['successful_flows'],
            'success_rate': (flow_counts['successful_flows'] / max(1, flow_counts['total_flows'])) * 100
        })
    
    # Create DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    csv_path = f"{output_dir}/flow_counts.csv"
    results_df.to_csv(csv_path, index=False)
    
    print(f"Flow counts saved to {csv_path}")
    
    # Print summary
    total_sent = results_df['total_flows_sent'].sum()
    total_successful = results_df['successful_flows'].sum()
    overall_success_rate = (total_successful / max(1, total_sent)) * 100
    
    print(f"\nSummary:")
    print(f"Total flows sent: {total_sent}")
    print(f"Total successful flows: {total_successful}")
    print(f"Overall success rate: {overall_success_rate:.2f}%")
    
    # Also create a summary CSV with aggregated stats
    summary_df = results_df.groupby('traffic_type').agg({
        'total_flows_sent': 'sum',
        'successful_flows': 'sum'
    }).reset_index()
    
    summary_df['success_rate'] = (summary_df['successful_flows'] / 
                                 summary_df['total_flows_sent']) * 100
    
    summary_csv_path = f"{output_dir}/flow_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Flow summary by traffic type saved to {summary_csv_path}")

def main(data_dir, output_dir):
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Usa il nome dell'ultima cartella di data_dir come exp_id
    exp_id = os.path.basename(os.path.abspath(data_dir))
    
    # Pattern per cercare i file CSV nella cartella dei dati
    bg_client_pattern = os.path.join(data_dir, "bg_client_*.csv")
    bursty_server_pattern = os.path.join(data_dir, "bursty_client_*.csv")
    
    # Cerca i file che matchano i pattern
    bg_client_files = glob.glob(bg_client_pattern)
    bursty_server_files = glob.glob(bursty_server_pattern)
    
    # Liste per accumulare i valori
    all_fct = []
    all_qct = []
    
    # Processa i file bg_client per estrarre la colonna "fct"
    for csv_file in bg_client_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                if "flow_completion_time" in row:
                    try:
                        all_fct.append(float(row["flow_completion_time"]))
                    except ValueError:
                        # Se il valore non è numerico, lo ignora
                        pass

    # Processa i file bursty_server per estrarre la colonna "qct"
    for csv_file in bursty_server_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                if "qct" in row:
                    try:
                        all_qct.append(float(row["qct"]))
                    except ValueError:
                        pass

    # Definisce i nomi dei file di output utilizzando l'exp_id
    fct_filename = os.path.join(output_dir, f"fct_{exp_id}")
    qct_filename = os.path.join(output_dir, f"qct_{exp_id}")

    # Scrive il file contenente i valori di FCT
    with open(fct_filename, "w", encoding="utf-8") as fout:
        for value in all_fct:
            fout.write(f"{value}\n")

    # Scrive il file contenente i valori di QCT
    with open(qct_filename, "w", encoding="utf-8") as fout:
        for value in all_qct:
            fout.write(f"{value}\n")
    
    fct_avg = sum(all_fct) / len(all_fct) if all_fct else 0
    qct_avg = sum(all_qct) / len(all_qct) if all_qct else 0

    print("Script completato.")
    print(f"Trovati {len(bg_client_files)} file bg_client e {len(bursty_server_files)} file bursty_client.")
    print(f"FCT estratti: {len(all_fct)}")
    print(f"QCT estratti: {len(all_qct)}")
    print(f"Output scritti in: {fct_filename} e {qct_filename}")
    #analyze_experiment_pcaps(data_dir, output_dir)
    print(f"Media FCT: {fct_avg:.5f}")
    print(f"Media QCT: {qct_avg:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estrae i dati 'fct' e 'qct' dai file CSV nella directory specificata e li scrive in output "
                    "utilizzando il nome della cartella di input come exp_id"
    )
    parser.add_argument("--data_dir", default=".", help="Directory contenente i file CSV (default: cartella corrente)")
    parser.add_argument("--output_dir", required=True, help="Directory in cui salvare i file di output")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
