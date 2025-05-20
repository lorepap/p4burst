#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import FuncFormatter

# Impostazioni generali per i grafici
plt.rcParams.update({'font.size': 14})
sns.set_style("whitegrid")
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6
CONFIDENCE_LEVEL = 0.95  # Intervallo di confidenza al 95%

# Mappatura di nomi policy (codici) a label più significative
POLICY_LABELS = {
    'dist_preemptive_deflection': 'Defl. Pred. Distribuita',
    'quantile_preemptive_deflection': 'Defl. Pred. Quantili',
    'simple_deflection': 'Deflessione Semplice',
    'ecmp': 'ECMP',
}


def calculate_confidence_interval(data, confidence=CONFIDENCE_LEVEL):
    """
    Calcola la metà ampiezza dell'intervallo di confidenza sui dati.
    """
    n = len(data)
    if n < 2:
        return 0.0
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return h


def load_data(data_dir):
    """
    Scorre ricorsivamente data_dir e raccoglie metriche per run.
    Ritorna DataFrame con colonne:
      policy, policy_label, fct, qct,
      deflected_packets_percentage, dropped_packets_percentage, lost_packets,
      sum_egress, sum_ingress, sum_reg_sum, sum_ing_reg_sum, sum_egr_reg_sum, 
      global_max_traversal_time_ns, global_max_ing_traversal_time_ns, global_max_egr_traversal_time_ns
    """
    runs = {}
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            path = os.path.join(root, fname)
            run = os.path.basename(root)
            policy = run.rsplit('_run', 1)[0]
            entry = runs.setdefault(run, {'policy': policy})

            if fname.startswith('fct_'):
                try:
                    vals = pd.read_csv(path, header=None).iloc[:, 0].astype(float).values
                    entry['fct'] = np.mean(vals)
                except Exception:
                    pass

            elif fname.startswith('qct_'):
                try:
                    vals = pd.read_csv(path, header=None).iloc[:, 0].astype(float).values
                    entry['qct'] = np.mean(vals)
                except Exception:
                    pass

            elif fname == 'switch_counters.csv':
                tc = pd.read_csv(path)
                total = tc['total'].sum()
                ingress_total = tc['ingress_total'].sum()
                egress = tc['egress'].sum()
                deflec = tc['deflected'].sum()
                dropped = tc['dropped'].sum()
                implicit = tc.get('implicitly_dropped', pd.Series()).sum()
                lost = total - (egress + implicit)
                entry['sum_egress'] = egress
                entry['sum_ingress'] = ingress_total
                entry['deflected_packets_percentage'] = deflec / total * 100
                entry['dropped_packets_percentage'] = dropped / total * 100
                entry['lost_packets'] = lost

            elif fname == 'traversal_summary.csv':
                ts = pd.read_csv(path)
                entry['sum_reg_sum'] = ts['total_reg_sum'].sum()
                entry['sum_ing_reg_sum'] = ts['total_ing_reg_sum'].sum()
                entry['sum_egr_reg_sum'] = ts['total_egr_reg_sum'].sum()
                entry['global_max_traversal_time_ns'] = ts['global_max_traversal_time_ns'].max()
                entry['global_max_ing_traversal_time_ns'] = ts['global_max_ing_traversal_time_ns'].max()
                entry['global_max_egr_traversal_time_ns'] = ts['global_max_egr_traversal_time_ns'].max()

    df = pd.DataFrame.from_dict(runs, orient='index')
    df = df.dropna(subset=[
        'fct', 'qct', 'deflected_packets_percentage',
        'dropped_packets_percentage', 'lost_packets',
        'sum_egress', 'sum_ingress', 'sum_reg_sum', 'sum_ing_reg_sum', 'sum_egr_reg_sum',
        'global_max_traversal_time_ns', 'global_max_ing_traversal_time_ns', 'global_max_egr_traversal_time_ns'
    ])
    df = df.reset_index(drop=True)
    df['policy_label'] = df['policy'].map(POLICY_LABELS).fillna(df['policy'])
    return df


def plot_metric_with_ci(df, metric, ylabel, title, out_file, percentage=False):
    labels = df['policy_label'].unique()
    means, errors = [], []
    for lbl in labels:
        vals = df[df['policy_label'] == lbl][metric].values
        means.append(np.mean(vals))
        errors.append(calculate_confidence_interval(vals))

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percentage:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
    for i, m in enumerate(means):
        txt = f"{m:.1f}%" if percentage else f"{m:.2f}"
        ax.text(i, m + (errors[i] if errors else 0) + 1e-6 * m,
                txt, ha='center', va='bottom')
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def plot_simple_bar(labels, values, ylabel, title, out_file, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    x = np.arange(len(labels))
    ax.bar(x, values, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v + 1e-6 * v, fmt.format(v), ha='center', va='bottom')
    plt.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Genera grafici comparativi per policy dai risultati di più run")
    parser.add_argument('data_dir', help="Directory contenente le sottocartelle dei run")
    args = parser.parse_args()
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        print(f"ERRORE: '{data_dir}' non è valida", file=sys.stderr)
        sys.exit(1)

    df = load_data(data_dir)

    # Grafici run-level
    run_metrics = [
        ('fct', 'Tempo completamento flusso (s)', 'Media FCT per policy (CI 95%)', 'fct_comparison.png', False),
        ('qct', 'Tempo completamento coda (s)', 'Media QCT per policy (CI 95%)', 'qct_comparison.png', False),
        ('dropped_packets_percentage', '% pacchetti scartati', '% scarti (CI 95%)', 'dropped_packets_comparison.png', True),
        ('deflected_packets_percentage', '% pacchetti deviati', '% deviazioni (CI 95%)', 'deflected_packets_comparison.png', True),
        ('lost_packets', 'Pacchetti persi', 'Pacchetti persi (CI 95%)', 'lost_packets_comparison.png', False)
    ]
    for metric, ylabel, title, fname, pct in run_metrics:
        plot_metric_with_ci(df, metric, ylabel, title, os.path.join(data_dir, fname), pct)

    # Aggregazione switch-level per policy
    agg = df.groupby('policy_label').agg({
        'sum_reg_sum': 'sum',
        'sum_ing_reg_sum': 'sum',
        'sum_egr_reg_sum': 'sum',
        'sum_egress': 'sum',
        'sum_ingress': 'sum',
        'global_max_traversal_time_ns': 'max',
        'global_max_ing_traversal_time_ns': 'max',
        'global_max_egr_traversal_time_ns': 'max'
    }).reset_index()
    # conversione in microsecondi per tempi medi
    agg['avg_switch_time_us'] = agg['sum_reg_sum'] / agg['sum_egress'] / 1e3
    agg['avg_ing_switch_time_us'] = agg['sum_ing_reg_sum'] / agg['sum_ingress'] / 1e3  # Usa il nuovo contatore
    agg['avg_egr_switch_time_us'] = agg['sum_egr_reg_sum'] / agg['sum_egress'] / 1e3
    # conversione in microsecondi per tempi massimi
    agg['max_traversal_us'] = agg['global_max_traversal_time_ns'] / 1e3
    agg['max_ing_traversal_us'] = agg['global_max_ing_traversal_time_ns'] / 1e3
    agg['max_egr_traversal_us'] = agg['global_max_egr_traversal_time_ns'] / 1e3

    labels = agg['policy_label'].tolist()
    
    # Plot tempo medio totale switch in μs
    plot_simple_bar(labels,
                   agg['avg_switch_time_us'].tolist(),
                   'Tempo medio switch (μs)',
                   'Tempo medio attraversamento switch per policy',
                   os.path.join(data_dir, 'avg_switch_time_comparison.png'),
                   fmt="{:.2f}")
    
    # Plot tempo medio ingress switch in μs
    plot_simple_bar(labels,
                   agg['avg_ing_switch_time_us'].tolist(),
                   'Tempo medio ingress (μs)',
                   'Tempo medio attraversamento ingress per policy',
                   os.path.join(data_dir, 'avg_ing_switch_time_comparison.png'),
                   fmt="{:.2f}")
    
    # Plot tempo medio egress switch in μs
    plot_simple_bar(labels,
                   agg['avg_egr_switch_time_us'].tolist(),
                   'Tempo medio egress (μs)',
                   'Tempo medio attraversamento egress per policy',
                   os.path.join(data_dir, 'avg_egr_switch_time_comparison.png'),
                   fmt="{:.2f}")
    
    # Plot max traversal in μs
    plot_simple_bar(labels,
                   agg['max_traversal_us'].tolist(),
                   'Tempo massimo switch (μs)',
                   'Tempo massimo attraversamento switch per policy',
                   os.path.join(data_dir, 'global_max_traversal_comparison.png'),
                   fmt="{:.2f}")
    
    # Plot max ingress traversal in μs
    plot_simple_bar(labels,
                   agg['max_ing_traversal_us'].tolist(),
                   'Tempo massimo ingress (μs)',
                   'Tempo massimo attraversamento ingress per policy',
                   os.path.join(data_dir, 'global_max_ing_traversal_comparison.png'),
                   fmt="{:.2f}")
    
    # Plot max egress traversal in μs
    plot_simple_bar(labels,
                   agg['max_egr_traversal_us'].tolist(),
                   'Tempo massimo egress (μs)',
                   'Tempo massimo attraversamento egress per policy',
                   os.path.join(data_dir, 'global_max_egr_traversal_comparison.png'),
                   fmt="{:.2f}")

    # Plot tempi medi di ingress vs egress in un grafico affiancato
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, agg['avg_ing_switch_time_us'].tolist(), width, label='Ingress')
    ax.bar(x + width/2, agg['avg_egr_switch_time_us'].tolist(), width, label='Egress')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Tempo medio attraversamento (μs)')
    ax.set_title('Confronto tempi medi ingress vs egress per policy')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(data_dir, 'ingress_vs_egress_time_comparison.png'))
    plt.close(fig)

    print("Grafici generati con successo!")

if __name__ == '__main__':
    main()
