#!/usr/bin/env python3
import subprocess
import csv
import os
import psutil
import signal
from datetime import datetime
from utils.stats import calculate_fct, calculate_qct

# --- Configurazione ---
OUTPUT_DIR = "./results/load"
POLICIES = [
    #'simple_deflection',
    'ecmp',
    #'dist_preemptive_deflection',
    #'quantile_preemptive_deflection'
]
N_RUNS = 10
COMMON_ARGS = [
    "sudo", "-E", "python3", "experiment_runner.py",
    "--duration", "40",
    "--n_hosts", "20",
    "--n_leaf", "2",
    "--n_spine", "2",
    "--bw", "100",
    "--delay", "0",
    "--n_clients", "10",
    "--n_servers", "10",
    "--flow_iat", "0.01",
    "--flow_size", "10000",
    "--bursty_reply_size", "100000",
    "--burst_interval", "0.1",
    "--burst_servers", "10",
    "--burst_clients", "4",
    "--queue_rate", "1000",
    "--queue_depth", "64",
    "--disable_logging",
    "--disable_pcap",
    "--exp_id"
]


def kill_other_python3():
    """
    Termina tutti i processi python3 tranne il processo corrente.
    """
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        pid = proc.info['pid']
        if pid == current_pid:
            continue
        name = proc.info['name'] or ''
        cmd = proc.info['cmdline'] or []
        if 'python3' in name or any('python3' in part for part in cmd):
            try:
                os.kill(pid, signal.SIGTERM)
            except (psutil.NoSuchProcess, PermissionError):
                pass


def run_experiments():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for policy in POLICIES:
        summary_csv = os.path.join(OUTPUT_DIR, f"{policy}.csv")
        with open(summary_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["run", "timestamp", "fct_avg", "qct_avg"])
            
            for run_idx in range(1, N_RUNS + 1):
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                exp_id = f"{policy}_run{run_idx}"
                data_dir = f"./tmp/{exp_id}"
                os.makedirs(data_dir, exist_ok=True)

                cmd = COMMON_ARGS + [exp_id, "--policy", policy]
                print(f"\n=== Policy: {policy} | Run: {run_idx}/{N_RUNS} ===")
                print("Eseguo:", " ".join(cmd))

                # Esegue ed aspetta il termine
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                # L'output è disponibile in result.stdout e result.stderr
                
                # Calcolo FCT e QCT
                fct_vals, fct_avg = calculate_fct(data_dir, OUTPUT_DIR)
                qct_vals, qct_avg = calculate_qct(data_dir, OUTPUT_DIR)

                writer.writerow([run_idx, timestamp, fct_avg, qct_avg])
                print(f"→ Run {run_idx} completata: FCT_avg={fct_avg:.5f}, QCT_avg={qct_avg:.5f}")

                # Termina eventuali python3 rimasti
                kill_other_python3()

    '''
    print(f"\n*** Riepilogo per policy '{policy}' salvato in {summary_csv} ***")
    print("\n=== Generazione grafici di confronto ===")
    plot_cmd = ["python3", "plot_policies_bars.py", "--data-dir", OUTPUT_DIR]
    print("Eseguo:", " ".join(plot_cmd))
    plot_result = subprocess.run(plot_cmd, check=True)
    print("Grafici generati con successo!")
    '''
if __name__ == "__main__":
    run_experiments()