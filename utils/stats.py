import os
import glob
import csv

def calculate_fct(data_dir, output_dir):
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Usa il nome dell'ultima cartella di data_dir come exp_id
    exp_id = os.path.basename(os.path.abspath(data_dir))
    
    # Pattern per cercare i file CSV nella cartella dei dati
    bg_client_pattern = os.path.join(data_dir, "bg_client_*.csv")
    
    # Cerca i file che matchano i pattern
    bg_client_files = glob.glob(bg_client_pattern)
    
    # Liste per accumulare i valori
    all_fct = []
    
    # Processa i file bg_client per estrarre la colonna "fct"
    for csv_file in bg_client_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                if "fct" in row:
                    try:
                        all_fct.append(float(row["fct"]))
                    except ValueError:
                        # Se il valore non è numerico, lo ignora
                        pass

    # Definisce i nomi dei file di output utilizzando l'exp_id
    fct_filename = os.path.join(output_dir, f"fct_{exp_id}")

    # Scrive il file contenente i valori di FCT
    with open(fct_filename, "w", encoding="utf-8") as fout:
        for value in all_fct:
            fout.write(f"{value}\n")
    
    fct_avg = sum(all_fct) / len(all_fct) if all_fct else 0

    print("Calcolo FCT completato.")
    print(f"Trovati {len(bg_client_files)} file bg_client.")
    print(f"FCT estratti: {len(all_fct)}")
    print(f"Output scritto in: {fct_filename}")
    print(f"Media FCT: {fct_avg:.5f}")
    
    return all_fct, fct_avg

def calculate_qct(data_dir, output_dir):
    # Crea la cartella di output se non esiste
    os.makedirs(output_dir, exist_ok=True)
    
    # Usa il nome dell'ultima cartella di data_dir come exp_id
    exp_id = os.path.basename(os.path.abspath(data_dir))
    
    # Pattern per cercare i file CSV nella cartella dei dati
    bursty_client_pattern = os.path.join(data_dir, "bursty_client_*.csv")
    
    # Cerca i file che matchano i pattern
    bursty_client_files = glob.glob(bursty_client_pattern)
    
    # Liste per accumulare i valori
    all_qct = []
    
    # Processa i file bursty_client per estrarre la colonna "qct"
    for csv_file in bursty_client_files:
        with open(csv_file, "r", newline="", encoding="utf-8") as fin:
            reader = csv.DictReader(fin)
            for row in reader:
                if "qct" in row:
                    try:
                        all_qct.append(float(row["qct"]))
                    except ValueError:
                        # Se il valore non è numerico, lo ignora
                        pass

    # Definisce i nomi dei file di output utilizzando l'exp_id
    qct_filename = os.path.join(output_dir, f"qct_{exp_id}")

    # Scrive il file contenente i valori di QCT
    with open(qct_filename, "w", encoding="utf-8") as fout:
        for value in all_qct:
            fout.write(f"{value}\n")
    
    qct_avg = sum(all_qct) / len(all_qct) if all_qct else 0

    print("Calcolo QCT completato.")
    print(f"Trovati {len(bursty_client_files)} file bursty_client.")
    print(f"QCT estratti: {len(all_qct)}")
    print(f"Output scritto in: {qct_filename}")
    print(f"Media QCT: {qct_avg:.5f}")
    
    return all_qct, qct_avg

