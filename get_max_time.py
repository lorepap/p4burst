import os
import csv
import glob
import argparse

def find_max_traversal(base_dir):
    max_value = 0
    max_file_path = ""
    
    # Cerca ricorsivamente tutti i file switch_counters.csv
    for root, _, _ in os.walk(base_dir):
        csv_files = glob.glob(os.path.join(root, "switch_counters.csv"))
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if 'max_traversal' in row:
                            try:
                                curr_value = float(row['max_traversal'])
                                if curr_value > max_value:
                                    max_value = curr_value
                                    max_file_path = csv_file
                            except (ValueError, TypeError):
                                # Ignora valori non numerici
                                pass
            except Exception as e:
                print(f"Errore nella lettura del file {csv_file}: {e}")
    
    return max_value, max_file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trova il massimo valore di max_traversal nei file switch_counters.csv")
    parser.add_argument("directory", help="Directory base da cui iniziare la ricerca")
    args = parser.parse_args()
    
    max_traversal, file_path = find_max_traversal(args.directory)
    
    print(f"Valore massimo di max_traversal: {max_traversal}")
    print(f"Trovato nel file: {file_path}")