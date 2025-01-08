# set random seed
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
import pandas as pd

from gurobipy import GRB

from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyepo.model.opt import optModel

import matplotlib.pyplot as plt

from pyepo import EPO


class EnergyManagementSystem(optModel):
    def __init__(self, P_r_data, P_l_data, T=24):
        """
        Args:
            T (int): Orizzonte di predizione.
            eta_c (float): Efficienza di carica della batteria.
            eta_d (float): Efficienza di scarica della batteria.
            sigma (float): Tasso di auto-scarica della batteria.
            P_s_max (float): Massima potenza scambiata con il sistema di accumulo.
            s_min (float): Minimo stato di carica della batteria.
            s_max (float): Massimo stato di carica della batteria.
            initial_s (float): Stato di carica iniziale della batteria.
            P_r_data (list[float]): Profilo di potenza prodotta dai generatori rinnovabili.
            P_l_data (list[float]): Profilo di carico di potenza.
        """

        # SITE 12

        self.T = T
        self.eta_c = 0.95
        self.eta_d = 0.95
        self.sigma = 0.0042
        self.P_s_max = 200
        self.s_max = 800
        self.s_min = 0.1 * self.s_max
        self.initial_s = 200
        self.T_s = 1

        self.P_r_data = P_r_data
        self.P_l_data = P_l_data

        # Aggiungi un parametro per i limiti di potenza della rete
        self.grid_power_min = -4000  
        self.grid_power_max = 4000   

        # sense
        self.modelSense = EPO.MINIMIZE
        
        super().__init__()

    def _getModel(self):
        m = ConcreteModel()

        # Variabili
        m.P_s_in = Var(range(self.T), bounds=(0, self.P_s_max))  # Potenza di carica
        m.P_s_out = Var(range(self.T), bounds=(0, self.P_s_max)) # Potenza di scarica
        m.s = Var(range(self.T + 1), bounds=(self.s_min, self.s_max))
        m.P_g = Var(range(self.T), within=Reals)

        m.eps = Var(range(self.T), within=NonNegativeReals)

        # Parametri
        m.P_r = Param(range(self.T), initialize={t: self.P_r_data[t] for t in range(self.T)})
        m.P_l = Param(range(self.T), initialize={t: self.P_l_data[t] for t in range(self.T)})

        # Vincoli
        m.initial_state = Constraint(expr=m.s[0] == self.initial_s)
        m.state_transition = Constraint(range(self.T), rule=lambda model, t: model.s[t + 1] == 
                               (1 - self.sigma) * model.s[t] + 
                               self.eta_c * self.T_s * model.P_s_in[t] - 
                               (1 / self.eta_d) * self.T_s * model.P_s_out[t])
        
        m.power_balance = Constraint(range(self.T), rule=lambda model, t: model.P_g[t] == 
                                 model.P_r[t] - model.P_l[t] - model.P_s_in[t] + model.P_s_out[t])
        
        # Limiti di potenza della rete (P_g)        
        m.grid_power_limit_min = Constraint(range(self.T), rule=lambda model, t: model.P_g[t] >= self.grid_power_min)
        m.grid_power_limit_max = Constraint(range(self.T), rule=lambda model, t: model.P_g[t] <= self.grid_power_max)

        # Variables for optimization
        variables = {
            'P_s_in': m.P_s_in,  # Variabile di potenza di carica della batteria
            'P_s_out': m.P_s_out,  # Variabile di potenza di scarica della batteria
            's': m.s,  # Stato di carica della batteria
            'P_g': m.P_g,  # Potenza scambiata con la griglia
            'eps': m.eps  # Variabile di penalità per deviazioni o limiti
        }
        return m, variables

    def setObj(self, c):
        # Rimuovi l'obiettivo esistente, se presente
        if hasattr(self._model, 'obj'):
            self._model.del_component('obj')
            
        # Assicurati che c sia convertito da euro/MWh a euro/kWh prima di usarlo nella funzione obiettivo
        converted_prices = [price / 1000 for price in c]
        self._model.obj = Objective(expr=sum(-converted_prices[t] * (self._model.P_r[t] - self._model.P_l[t] -  
                                    self._model.P_s_in[t] + self._model.P_s_out[t]) +
                                    10000 * self._model.eps[t] for t in range(self.T)), sense=minimize)

    def solve(self):
        solver = SolverFactory('gurobi')
        results = solver.solve(self._model, tee=True)

        if results.solver.status == 'ok' and results.solver.termination_condition == 'optimal':
            # Salva i valori ottimizzati come attributi della classe
            self.P_g_values = np.array([value(self._model.P_g[t]) for t in range(self.T)])
            self.P_s_in_values = np.array([value(self._model.P_s_in[t]) for t in range(self.T)])  # Valori di potenza di carica
            self.P_s_out_values = np.array([value(self._model.P_s_out[t]) for t in range(self.T)])  # Valori di potenza di scarica
            self.s_values = np.array([value(self._model.s[t]) for t in range(self.T + 1)])  # Include lo stato finale
            self.eps_values = np.array([value(self._model.eps[t]) for t in range(self.T)])  # Valori delle penalità

            # Salva anche il valore dell'obiettivo
            self.objective_value = value(self._model.obj)

            # Crea un DataFrame con i risultati
            df_results = pd.DataFrame({
                'P_g': self.P_g_values,
                'P_s_in': self.P_s_in_values,
                'P_s_out': self.P_s_out_values,
                's': self.s_values[:-1],  # escludere l'ultimo stato per allineare le lunghezze delle colonne
                'eps': self.eps_values
                                                        })

            # Salva i risultati in un file CSV
            df_results.to_csv('results.csv', index=False)

            # Restituisci i valori ottimizzati di P_g e il valore dell'obiettivo per compatibilità
            return self.P_g_values, self.objective_value
        else:
            raise Exception("Non è stata trovata una soluzione ottimale. Stato: {}".format(results.solver.status))
        
    def plot_variables(self):
        # Assicurati che solve sia stato chiamato e che i valori siano disponibili
        if not hasattr(self, 'P_g_values') or not hasattr(self, 'P_s_in_values') or not hasattr(self, 'P_s_out_values') or not hasattr(self, 's_values') or not hasattr(self, 'eps_values'):
            raise ValueError("Il metodo 'solve' deve essere chiamato prima di 'plot_variables'.")

        # Creazione di un array temporale
        time = list(range(self.T))  # Per le variabili con dimensione T
        time_s = list(range(self.T + 1))  # Per s_values che include lo stato iniziale e finale

        # Plot delle variabili
        plt.figure(figsize=(14, 8))

        # Plot di P_s_in
        plt.subplot(2, 1, 1)
        plt.plot(time, self.P_s_in_values, label="Potenza di Carica (P_s_in)", marker='o', color='blue')

        # Plot di P_s_out
        plt.plot(time, self.P_s_out_values, label="Potenza di Scarica (P_s_out)", marker='x', color='red')

        # Plot di s
        plt.plot(time_s, self.s_values, label="Stato di Carica (s)", marker='s', color='green')  # Usa il tempo corretto

        # Plot di P_g
        plt.plot(time, self.P_g_values, label="Potenza alla Rete (P_g)", marker='^', color='purple')

        # Etichette e Titolo
        plt.xlabel("Tempo (ore)")
        plt.ylabel("Valore delle Variabili")
        plt.title("Andamento delle Variabili nel Tempo")
        plt.legend()
        plt.grid()

        # Plot delle penalità eps
        plt.subplot(2, 1, 2)
        plt.plot(time, self.eps_values, label="Penalità (eps)", marker='v', color='orange')
        plt.xlabel("Tempo (ore)")
        plt.ylabel("Valore delle Penalità")
        plt.title("Andamento delle Penalità nel Tempo")
        plt.legend()
        plt.grid()

        # Mostra il grafico
        plt.tight_layout()
        plt.show()

# Leggi i dati dai file CSV
pv_data = pd.read_csv('./dataset.csv', usecols=[1]).squeeze()
load_data = pd.read_csv('./dataset.csv', usecols=[2]).squeeze()
price_data = pd.read_csv('./dataset.csv', usecols=[3]).squeeze()


# DATAFRAME CON I DATI

df = pd.DataFrame({
    'PV Data': pv_data.values,
    'Load Data': load_data.values,
    'Price Data': price_data.values
})

"""
🎈🎈🎈🎈🎈 CALCOLO SU INTERO DATASET
# Inizializzazione del costo totale
costo_totale = 0

# Numero totale di ore nei dati
total_hours = df.shape[0]

# Iterazione attraverso ciascuna finestra di 24 ore
for start in range(0, total_hours, 24):
    end = start + 24
    if end > total_hours:
        break  # Se non ci sono abbastanza dati per una finestra completa, interrompiamo il ciclo

    # Estrazione dei dati per la finestra corrente
    P_r = df['PV Data'][start:end].tolist()
    P_l = df['Load Data'][start:end].tolist()
    price = df['Price Data'][start:end].tolist()

    # Creazione del modello per la finestra corrente
    model = EnergyManagementSystem(P_r, P_l, T=24)
    model.setObj(price)

    # Risoluzione del modello e aggiunta del costo alla somma totale
    _, costo_finestra = model.solve()
    costo_totale += costo_finestra

# Stampa del costo totale
print(f"COSTO TOTALE: {costo_totale:.2f} EURO")
"""


# Imposta i dati di generazione e carico nel modello
P_r = df['PV Data'].tolist()
P_l = df['Load Data'].tolist()
price = df['Price Data'].tolist()


model = EnergyManagementSystem(P_r[:24], P_l[:24], T = 24) 

#🎈🎈🎈🎈🎈🎈 SOLUZIONE PER UNA FINESTRA
# Imposta l'obiettivo di ottimizzazione
model.setObj(price[:24])
sol, obj = model.solve()

print(f"COSTO: {obj:.2f} EURO")


# Definizione della funzione per creare le finestre
def create_windows(data, lookback, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+forecast_horizon])
    return np.array(X), np.array(y)

# Creazione delle finestre di dati
lookback = 168
forecast_horizon = 24




# Usiamo tutte e tre le colonne per le features
features, _ = create_windows(df.values, lookback, forecast_horizon)

# reshape features da 3D array a 2D array per darlo in input all'MLP
features = features.reshape(features.shape[0], -1)

# Usiamo solo i dati di prezzo per i target
_, price_windows = create_windows(df['Price Data'].values, lookback, forecast_horizon)

# Stampiamo le dimensioni per confermare che tutto sia corretto
print("Dimensioni delle features:", features.shape)        # Dovrebbe essere (num_windows, lookback*num_features)
print("Dimensioni delle price_windows:", price_windows.shape)  # Dovrebbe essere (num_windows, forecast_horizon)





from sklearn.model_selection import train_test_split


# Specifica la proporzione del set di test; ad esempio, 0.2 per il 20% come test set
test_size = 0.2

# Divide i dati in training e test set senza mescolare
features_train, features_test, price_windows_train, price_windows_test = train_test_split(
    features, 
    price_windows, 
    test_size=test_size, 
    shuffle=False  # Importante per mantenere l'ordine temporale
)

# Stampiamo le dimensioni per verificare
print("Dimensioni features_train:", features_train.shape)
print("Dimensioni features_test:", features_test.shape)
print("Dimensioni price_windows_train:", price_windows_train.shape)
print("Dimensioni price_windows_test:", price_windows_test.shape)



import pyepo
from pyepo.data.dataset import optDataset

# get training data set
dataset_train = optDataset(model, features_train, price_windows_train)
# get test data set
dataset_test = optDataset(model, features_test, price_windows_test)

# get data loader
from torch.utils.data import DataLoader
batch_size = 4
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)


import torch
from torch import nn

class multiLayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(multiLayerPerceptron, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# model size
input_dim = features.shape[-1]
output_dim = price_windows.shape[-1]
hidden_dim = 48
# init for test
reg = multiLayerPerceptron(input_dim, hidden_dim, output_dim)

import time
from tqdm import tqdm
from pyepo.metric import regret

def trainModel(reg, func, method_name, loader_train, loader_test, optmodel,
                device="cpu", lr=1e-1, num_epochs=1):
    """
    Train a model using PyEPO.
    """
    # set optimizer
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    # init loss functions
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    # set model to training mode
    reg.train()
    # init log
    loss_log, regret_log_train, regret_log_test = [], [], []
    tbar = tqdm(range(num_epochs))
    elapsed = 0
    for epoch in tbar:
        # eval
        regret_train = regret(reg, optmodel, loader_train)
        regret_log_train.append(regret_train)
        regret_test = regret(reg, optmodel, loader_test)
        regret_log_test.append(regret_test)
        # record time elapsed for training
        tick = time.time()
        # iterate over data mini-batches
        for i, data in enumerate(loader_train):
            # load data
            x, c, w, z = data
            # send to device
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            print("x type:", type(x), "x shape:", x.shape)
            print("c type:", type(c), "c shape:", c.shape)
            print("w type:", type(w), "w shape:", w.shape)
            print("z type:", type(z), "z shape:", z.shape)


            # FORWARD PASS
            cp = reg(x) # prediction
            if method_name == "SPO+":
                # spo+ loss
                loss = func(cp, c, w, z)
            else:
                raise ValueError("Unknown method_name: {}".format(method_name))
            # regularization term
            #loss += 10 * l2(cp, c)


            # BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss
            loss_log.append(loss.item())
            # update progress bar
            tbar.set_description("Epoch: {:2}, Loss: {:3.4f}".format(epoch, loss.item()))
            # elapsed time
            tock = time.time()
            elapsed += tock - tick

    # eval
    regret_train = regret(reg, optmodel, loader_train)
    regret_log_train.append(regret_train)
    regret_test = regret(reg, optmodel, loader_test)
    regret_log_test.append(regret_test)
    # save logs
    logs = {}
    logs["loss"] = np.array(loss_log)
    logs["regret_train"] = np.array(regret_log_train)
    logs["regret_test"] = np.array(regret_log_test)
    logs["elapsed"] = elapsed
    # final result
    print("{}: Regret on test set: {:.2f}% after {:.2f} sec of training".format(method, regret_test*100, elapsed))
    return logs

num_processes = 1 # number of cores (0 --> all cores)
num_epochs = 1    # number of epochs
lr = 1e-2         # learning rate
device = "cpu"    # device to use


from pyepo.func import SPOPlus

method = 'SPO+'
func = SPOPlus(model, processes=num_processes)

print("Method:", method)
# init model
# reg = LinearRegressionNN()
reg = multiLayerPerceptron(input_dim, hidden_dim, output_dim)
# training
logs = trainModel(reg, func, method, loader_train, loader_test, model,
                    device=device, lr=lr, num_epochs=num_epochs)
# eval
regret_test = regret(reg, model, loader_test)

"""# draw plot
plotLearningCurve(logs, method)
visSol(plo_t, pup_t, loader_test, optmodel, ind=10, pytorch_model=reg, method_name=method)
"""