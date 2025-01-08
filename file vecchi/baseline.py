import numpy as np
import pandas as pd

def baseline_objective():
    # Stessi indici
    T = [t for t in range(24)]

    # Parametri di costo di acquisto e vendita
    PR_buy = [155 for _ in T]
    PR_sell = [152.5 for _ in T]

    # Caricamento e normalizzazione dati PV
    pv_ref = 500
    pv_data = pd.read_csv('miris_pv.csv').to_numpy()
    pv_data = pv_data[:17280, 1]
    pv_data = np.mean(pv_data.reshape(-1, 720), axis=1)
    pv_data = pv_ref * pv_data / np.amax(pv_data)
    P_pv = pv_data.tolist()

    # Caricamento e normalizzazione del carico
    load_fix_ref = 2000
    load_fix_data = pd.read_csv('miris_load.csv').to_numpy()
    load_fix_data = load_fix_data[:17280, 1]
    load_fix_data = np.mean(load_fix_data.reshape(-1, 720), axis=1)
    load_fix_data = load_fix_ref * load_fix_data / np.amax(load_fix_data)
    P_L_fix = load_fix_data.tolist()

    # Niente generatori convenzionali (CDG)
    # Niente batteria
    # Nessun load shifting
    # Si acquista o vende energia da/alla rete per coprire la differenza tra load e PV
    # L'obiettivo è: costo delle transazioni (acquisto - vendita)
    # Dato che i costi CDG e le penalità di shifting sono nulli in questo baseline

    transaction_cost = 0.0
    for t in T:
        surplus = max(0, P_pv[t] - P_L_fix[t])
        shortage = max(0, P_L_fix[t] - P_pv[t])
        transaction_cost += PR_buy[t] * shortage - PR_sell[t] * surplus

    return transaction_cost

obj_baseline = baseline_objective()
print(obj_baseline)
