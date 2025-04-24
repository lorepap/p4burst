#!/usr/bin/env python3
import subprocess
import csv
import os
import psutil
import signal
import math
import statistics
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
from utils.stats import calculate_fct, calculate_qct

# --- Configurazione ---
OUTPUT_DIR = "./results/vary_load"
POLICIES = [
    'simple_deflection',
    'ecmp',
    'dist_preemptive_deflection',
    'quantile_preemptive_deflection'
]
# Livelli di load di background misurati come numero di client simultanei
LOAD_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40]
N_RUNS = 5  # numero di run per combinazione

COMMON_ARGS = [
    "sudo", "-E", "python3", "experiment_runner.py",
    "--duration", "30",
    "--n_hosts", "40",          # per permettere fino a 40 client
    "--n_leaf", "4",
    "--n_spine", "2",
    "--bw", "10",
    "--delay", "0.0001",
    # "--n_clients" verrà inserito dinamicamente
    "--n_servers", "10",
    "--flow_iat", "0.001",
    "--flow_size", "1000",
    "--bursty_reply_size", "4000",
    "--burst_interval", "0.05",
    "--burst_servers", "20",
    "--burst_clients", "10",
    "--queue_rate", "10",
    "--queue_depth", "10",
    "--exp_id"
]

def kill_other_python3():
    """Termina tutti i processi python3 tranne il processo corrente."""
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
    master_csv = os.path.join(OUTPUT_DIR, "aggregate_results.csv")
    with open(master_csv, "w", newline="") as aggfile:
        writer = csv.writer(aggfile)
        writer.writerow(["policy", "load", "run", "timestamp", "fct_avg", "qct_avg"])

        for load in LOAD_LEVELS:
            print(f"\n=== Load di background: {load} client ===")
            for policy in POLICIES:
                for run_idx in range(1, N_RUNS + 1):
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    exp_id = f"{policy}_load{load}_run{run_idx}"
                    data_dir = f"./tmp/{exp_id}"
                    os.makedirs(data_dir, exist_ok=True)

                    cmd = (
                        COMMON_ARGS + [exp_id, "--n_clients", str(load), "--policy", policy]
                    )
                    print(f"Eseguo: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                    # calcolo metriche
                    fct_vals, fct_avg = calculate_fct(data_dir, OUTPUT_DIR)
                    qct_vals, qct_avg = calculate_qct(data_dir, OUTPUT_DIR)

                    writer.writerow([policy, load, run_idx, timestamp, fct_avg, qct_avg])
                    print(f"→ {policy} @ load={load} run {run_idx}: FCT={fct_avg:.5f}, QCT={qct_avg:.5f}")

                    kill_other_python3()

    print(f"\nDati aggregati salvati in {master_csv}")
    return master_csv

def generate_plots(data_csv):
    # Leggi dati aggregati
    data = {}
    loads = set()
    with open(data_csv, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pol = row['policy']
            load = int(row['load'])
            fct = float(row['fct_avg'])
            qct = float(row['qct_avg'])
            loads.add(load)
            data.setdefault(pol, {}).setdefault(load, []).append((fct, qct))

    loads = sorted(loads)
    policies = sorted(data.keys())
    ALPHA = 0.05

    # Calcola medie e intervalli di confidenza
    stats_pol = {pol: {'fct_mean': [], 'fct_ci': [], 'qct_mean': [], 'qct_ci': []}
                 for pol in policies}

    for pol in policies:
        for load in loads:
            vals = data[pol].get(load, [])
            fcts = [v[0] for v in vals]
            qcts = [v[1] for v in vals]
            n = len(fcts)

            mu_f = statistics.mean(fcts)
            sd_f = statistics.stdev(fcts) if n > 1 else 0
            t_crit = stats.t.ppf(1 - ALPHA/2, df=n-1) if n > 1 else 0
            ci_f = t_crit * sd_f / math.sqrt(n) if n > 1 else 0

            mu_q = statistics.mean(qcts)
            sd_q = statistics.stdev(qcts) if n > 1 else 0
            ci_q = t_crit * sd_q / math.sqrt(n) if n > 1 else 0

            stats_pol[pol]['fct_mean'].append(mu_f)
            stats_pol[pol]['fct_ci'].append(ci_f)
            stats_pol[pol]['qct_mean'].append(mu_q)
            stats_pol[pol]['qct_ci'].append(ci_q)

    # Directory output plot
    out_dir = os.path.dirname(data_csv)

    # Plot FCT vs Load
    plt.figure()
    for pol in policies:
        x = loads
        y = stats_pol[pol]['fct_mean']
        ci = stats_pol[pol]['fct_ci']
        plt.plot(x, y, marker='o', label=pol)
        lower = [yi - ci_i for yi, ci_i in zip(y, ci)]
        upper = [yi + ci_i for yi, ci_i in zip(y, ci)]
        plt.fill_between(x, lower, upper, alpha=0.2)
    plt.xlabel("Load (n_clients)")
    plt.ylabel("FCT medio")
    plt.title("FCT vs Load di background")
    plt.xticks(loads)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fct_plot = os.path.join(out_dir, "fct_vs_load_line.png")
    plt.savefig(fct_plot)
    plt.close()

    # Plot QCT vs Load
    plt.figure()
    for pol in policies:
        x = loads
        y = stats_pol[pol]['qct_mean']
        ci = stats_pol[pol]['qct_ci']
        plt.plot(x, y, marker='s', label=pol)
        lower = [yi - ci_i for yi, ci_i in zip(y, ci)]
        upper = [yi + ci_i for yi, ci_i in zip(y, ci)]
        plt.fill_between(x, lower, upper, alpha=0.2)
    plt.xlabel("Load (n_clients)")
    plt.ylabel("QCT medio")
    plt.title("QCT vs Load di background")
    plt.xticks(loads)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    qct_plot = os.path.join(out_dir, "qct_vs_load_line.png")
    plt.savefig(qct_plot)
    plt.close()

    print("\nGrafici generati:")
    print(f" - {fct_plot}")
    print(f" - {qct_plot}")

if __name__ == "__main__":
    csv_path = run_experiments()
    generate_plots(csv_path)
