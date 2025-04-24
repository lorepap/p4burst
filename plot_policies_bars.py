import os
import csv
import math
import statistics
from scipy import stats
import matplotlib.pyplot as plt
import argparse

# Configurazione argparse per prendere BASE_DATA_DIR da linea di comando
parser = argparse.ArgumentParser(description='Genera grafici a barre per confrontare le politiche.')
parser.add_argument('--data-dir', type=str, default='./results/load',
                    help='Directory contenente i file CSV dei risultati (default: ./results/load)')
args = parser.parse_args()

# Usa il valore da linea di comando
BASE_DATA_DIR = args.data_dir

POLICIES = [
    'ecmp',
    'simple_deflection',
    'dist_preemptive_deflection',
    'quantile_preemptive_deflection'
]
LABELS = [
    'ECMP',
    'Simple D.',
    'Dist. Preemptive D.',
    'Quant Preemptive D.'
]
ALPHA = 0.05  # per intervallo di confidenza al 95%

# Liste per memorizzare risultati
fct_means = []
fct_cis = []
qct_means = []
qct_cis = []

# Calcolo medie e intervalli di confidenza per ogni policy
for policy in POLICIES:
    summary_path = os.path.join(BASE_DATA_DIR, f"{policy}.csv")
    fct_vals, qct_vals = [], []
    
    with open(summary_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fct_vals.append(float(row["fct_avg"]))
            qct_vals.append(float(row["qct_avg"]))
    
    n = len(fct_vals)
    # FCT
    mu_fct = statistics.mean(fct_vals)
    sd_fct = statistics.stdev(fct_vals)
    t_crit = stats.t.ppf(1 - ALPHA/2, df=n-1)
    margin_fct = t_crit * sd_fct / math.sqrt(n)
    fct_means.append(mu_fct)
    fct_cis.append(margin_fct)
    # QCT
    mu_qct = statistics.mean(qct_vals)
    sd_qct = statistics.stdev(qct_vals)
    margin_qct = t_crit * sd_qct / math.sqrt(n)
    qct_means.append(mu_qct)
    qct_cis.append(margin_qct)

# Creazione directory plots
os.makedirs(BASE_DATA_DIR, exist_ok=True)

# Plot FCT e salvataggio
plt.figure()
x = range(len(POLICIES))
plt.bar(x, fct_means, yerr=fct_cis, capsize=5)
plt.xticks(x, LABELS, rotation=45, ha='right')
plt.ylabel("FCT medio")
plt.title("Confronto FCT con intervalli di confidenza al 95%")
plt.tight_layout()
fct_plot_path = os.path.join(BASE_DATA_DIR, "fct_comparison.png")
plt.savefig(fct_plot_path)
plt.close()

# Plot QCT e salvataggio
plt.figure()
plt.bar(x, qct_means, yerr=qct_cis, capsize=5)
plt.xticks(x, LABELS, rotation=45, ha='right')
plt.ylabel("QCT medio")
plt.title("Confronto QCT con intervalli di confidenza al 95%")
plt.tight_layout()
qct_plot_path = os.path.join(BASE_DATA_DIR, "qct_comparison.png")
plt.savefig(qct_plot_path)
plt.close()

print(f"Plots salvati in:\n - {fct_plot_path}\n - {qct_plot_path}")
