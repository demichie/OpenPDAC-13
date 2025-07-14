#!/bin/bash

# Termina lo script immediatamente se un comando fallisce
set -e

# --- INIZIO FUNZIONE DI MODIFICA ---
# Funzione intelligente che modifica il file di controllo corretto per un'esecuzione rapida in CI.
# Priorità:
# 1. Modifica 'system/controlDict.run' se esiste.
# 2. Altrimenti, modifica 'system/controlDict' se esiste.
#
# Argomenti:
# $1: Nuovo valore per endTime
# $2: Nuovo valore per writeInterval
shorten_run() {
  local new_end_time=$1
  local new_write_interval=$2
  local target_dict=""

  echo "CI environment detected. Shortening run for this test case."

  # Determina quale file di dizionario modificare
  if [ -f "system/controlDict.run" ]; then
    target_dict="system/controlDict.run"
    echo "Found 'system/controlDict.run'. This will be modified."
  elif [ -f "system/controlDict" ]; then
    target_dict="system/controlDict"
    echo "Found 'system/controlDict'. This will be modified."
  else
    echo "ERROR: Could not find 'system/controlDict.run' or 'system/controlDict' to modify."
    exit 1
  fi

  # Esegui la modifica sul file target
  echo "--> Modifying ${target_dict}: setting endTime=${new_end_time}, writeInterval=${new_write_interval}"
  
  # Fai un backup prima di modificare
  cp "$target_dict" "${target_dict}.bak"
  
  # Usa 'sed' per modificare endTime e writeInterval
  sed -i "s/^endTime .*/endTime         ${new_end_time};/" "$target_dict"
  sed -i "s/^writeInterval .*/writeInterval   ${new_write_interval};/" "$target_dict"
}
# --- FINE FUNZIONE DI MODIFICA ---

# Itera su tutte le sottocartelle della directory corrente ('run')
for test_case_dir in synthTopo2D/; do
  test_case_dir=${test_case_dir%/}

  if [ -f "${test_case_dir}/.noTest" ]; then
    echo "----------------------------------------------------"
    echo "Skipping test case: ${test_case_dir} (.noTest file found)"
    echo "----------------------------------------------------"
    continue # Questo è il comando chiave per saltare l'iterazione corrente del loop
  fi

  echo "----------------------------------------------------"
  echo "Processing test case: ${test_case_dir}"
  echo "----------------------------------------------------"
  
  # Esegue i comandi per ogni test in una sub-shell per isolare l'ambiente
  (
    cd "${test_case_dir}"
    
    # Se siamo in CI e il file marcatore esiste, accorcia la simulazione
    if [ "$CI" == "true" ] && [ -f .shorten_in_ci ]; then
      # Chiama la funzione per accorciare la run.
      # Imposta un tempo finale breve e un intervallo di scrittura che garantisca almeno un output.
      shorten_run "0.01" "2" 
    fi

    # Esegui Allrun usando foamRun per catturare il log
    if [ -f Allrun ]; then
      echo "Running simulation script (Allrun)..."
      chmod +x ./Allrun
      ./Allrun
    else
      echo "ERROR: Allrun script not found!"
      exit 1
    fi

    # Esegui la validazione con Alltest
    if [ -f Alltest ]; then
      echo "Validating results (Alltest)..."
      chmod +x ./Alltest
      ./Alltest
      ./Allclean      
    else
      echo "WARNING: Alltest script not found for ${test_case_dir}. Skipping validation."
    fi

    # Ripristina i file di dizionario dai backup
    for bak_file in system/*.bak; do
      if [ -f "$bak_file" ]; then
        original_file="${bak_file%.bak}"
        mv "$bak_file" "$original_file"
      fi
    done
  )
done

echo "----------------------------------------------------"
echo "All test cases processed successfully!"
echo "----------------------------------------------------"
