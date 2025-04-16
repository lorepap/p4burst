#!/usr/bin/env python3

import os
import glob
import csv
import argparse

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

    print("Script completato.")
    print(f"Trovati {len(bg_client_files)} file bg_client e {len(bursty_server_files)} file bursty_client.")
    print(f"FCT estratti: {len(all_fct)}")
    print(f"QCT estratti: {len(all_qct)}")
    print(f"Output scritti in: {fct_filename} e {qct_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estrae i dati 'fct' e 'qct' dai file CSV nella directory specificata e li scrive in output "
                    "utilizzando il nome della cartella di input come exp_id"
    )
    parser.add_argument("--data_dir", default=".", help="Directory contenente i file CSV (default: cartella corrente)")
    parser.add_argument("--output_dir", required=True, help="Directory in cui salvare i file di output")
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
