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
    def __init__(self, T=24):
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

        # Aggiungi un parametro per i limiti di potenza della rete
        self.grid_power_min = -4000  
        self.grid_power_max = 4000   

        # sense
        self.modelSense = EPO.MINIMIZE
        
        super().__init__()


    def update_parameters(self, P_r, P_l):
        """
        Update the parameters P_r and P_l for the optimization model.

        Args:
            P_r (np.ndarray): Array of renewable power generation data.
            P_l (np.ndarray): Array of load demand data.
        """

        for t in range(self.T):
            self._model.P_r[t].set_value(P_r[t])
            self._model.P_l[t].set_value(P_l[t])

    def _getModel(self):
        m = ConcreteModel()

        # Variabili
        m.P_s_in = Var(range(self.T), bounds=(0, self.P_s_max))  # Potenza di carica
        m.P_s_out = Var(range(self.T), bounds=(0, self.P_s_max)) # Potenza di scarica
        m.s = Var(range(self.T + 1), bounds=(self.s_min, self.s_max))
        m.P_g = Var(range(self.T), within=Reals)

        m.eps = Var(range(self.T), within=NonNegativeReals)

        # Parametri (inizializzati con valori placeholder)
        m.P_r = Param(range(self.T), mutable=True, initialize={t: 0 for t in range(self.T)})
        m.P_l = Param(range(self.T), mutable=True, initialize={t: 0 for t in range(self.T)})

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
            
        # Inglobo il segno meno della formula nel prezzo c    
        c = [-price for price in c]
        self._model.obj = Objective(expr=sum(c[t] * self._model.P_g[t] 
                                             + 10000 * self._model.eps[t] 
                                    for t in range(self.T)), sense=minimize)

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

price = df['Price Data']

# Converto da euro/MWh a euro/kWh prima di usarlo nella funzione obiettivo
price = price / 1000

model = EnergyManagementSystem(T = 24) 

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

# Creo le feature da dare alla rete neurale per la previsione (prezzi, + eventuali variabili esogene)
input_features, _ = create_windows(price.values[:200], lookback, forecast_horizon)

# Usiamo solo i dati di prezzo per i target
_, target_price = create_windows(price.values[:200], lookback, forecast_horizon)

# Stampiamo le dimensioni per confermare che tutto sia corretto
print("Dimensioni delle features:", input_features.shape)        # Dovrebbe essere (num_windows, lookback*num_features)
print("Dimensioni delle target_price:", target_price.shape)  # Dovrebbe essere (num_windows, forecast_horizon)


from sklearn.model_selection import train_test_split

# Specifica la proporzione del set di test; ad esempio, 0.2 per il 20% come test set
test_size = 0.2

# Divide i dati in training e test set senza shuffle
input_features_train, input_features_test, target_price_train, target_price_test = train_test_split(
    input_features, 
    target_price, 
    test_size=test_size, 
    shuffle=False 
)

from sklearn.preprocessing import MinMaxScaler

# Inizializzazione degli scaler
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# Fit degli scaler sui dati di training
scaler_features.fit(input_features_train)
scaler_target.fit(target_price_train)

"""# Trasformazione dei dati di training
input_features_train = scaler_features.transform(input_features_train)
target_price_train = scaler_target.transform(target_price_train)

# Trasformazione dei dati di test
input_features_test = scaler_features.transform(input_features_test)
target_price_test = scaler_target.transform(target_price_test)"""


input_features_train = input_features_train.reshape(input_features_train.shape[0], lookback, -1)
input_features_test = input_features_test.reshape(input_features_test.shape[0], lookback, -1)


# Stampiamo le dimensioni per verificare
print("Dimensioni features_train:", input_features_train.shape)
print("Dimensioni features_test:", input_features_test.shape)
print("Dimensioni target_price_train:", target_price_train.shape)
print("Dimensioni target_price_test:", target_price_test.shape)


import pyepo
from optdataset_modificato import optDataset

# 🎈🎈🎈🎈🎈🎈🎈🎈 Carico optdataset con modello, input, target, e parametri (Pl e pr) 🎈🎈🎈🎈🎈🎈🎈🎈

P_r = df['PV Data']
P_l = df['Load Data']

# Creo finestre di Pl e Pr contenenti T = forecast horizon elementi 
_, P_r_windows = create_windows(P_r.values[:200], lookback, forecast_horizon)
_, P_l_windows = create_windows(P_l.values[:200], lookback, forecast_horizon)

P_r_windows_train, P_r_windows_test, P_l_windows_train, P_l_windows_test = train_test_split(
    P_r_windows, 
    P_l_windows, 
    test_size=test_size, 
    shuffle=False 
)

# Inizializzazione degli scaler
scaler_P_r = MinMaxScaler()
scaler_P_l= MinMaxScaler()

# Fit degli scaler sui dati di training
scaler_P_r.fit(P_r_windows_train)
scaler_P_l.fit(P_l_windows_train)

"""# Trasformazione dei dati di training
P_r_windows_train = scaler_P_r.transform(P_r_windows_train)
P_l_windows_train = scaler_P_l.transform(P_l_windows_train)

# Trasformazione dei dati di test
P_r_windows_test = scaler_P_r.transform(P_r_windows_test)
P_l_windows_test = scaler_P_l.transform(P_l_windows_test)"""


input_features_train = input_features_train.reshape(input_features_train.shape[0], lookback, -1)
input_features_test = input_features_test.reshape(input_features_test.shape[0], lookback, -1)

# get training data set
dataset_train = optDataset(model, input_features_train, target_price_train, P_r_windows_train, P_l_windows_train)
# get test data set
dataset_test = optDataset(model, input_features_test, target_price_test, P_r_windows_test, P_l_windows_test)

# get data loader
from torch.utils.data import DataLoader
batch_size = 2
# batch_size_test = 1 # in origine era 1
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


##### 🎈🎈🎈🎈🎈🎈 DEFINIZIONE LSTM 🎈🎈🎈🎈🎈🎈🎈
import torch
from torch import nn

import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.linear(out)
        return out


# Parametri del modello
input_dim = 1  # Dimensione delle features
hidden_dim = 48  # Dimensione dello stato nascosto
output_dim = target_price.shape[-1]  # Numero di step di previsione
num_layers = 1  # Numero di layer LSTM

# Inizializzazione del modello
reg = LSTMModel(input_dim, hidden_dim, output_dim, num_layers)



def normalize_input(x, scaler):
    """Normalizza il tensore di input x utilizzando lo scaler fittato."""
    # Converti il tensore PyTorch in un array NumPy
    x_np = x.numpy()
    # Flattening delle prime due dimensioni per rispecchiare il formato usato durante il fitting dello scaler
    original_shape = x_np.shape
    x_reshaped = x_np.reshape(-1, original_shape[1] * original_shape[2])
    
    # Applica lo scaler
    x_scaled = scaler.transform(x_reshaped)
    
    # Ritorna alla forma originale (batch_size, lookback, num_features)
    x_scaled = x_scaled.reshape(original_shape)
    
    # Converti in un tensore PyTorch
    return torch.from_numpy(x_scaled).float()


def denormalize_output(cp, scaler):
    """Denormalizza le previsioni cp utilizzando lo scaler fittato per i target."""
    # Ottieni i parametri dello scaler
    min_val = torch.tensor(scaler.data_min_, device=cp.device, dtype=cp.dtype)
    scale = torch.tensor(scaler.scale_, device=cp.device, dtype=cp.dtype)

    # Applica la denormalizzazione
    cp_denormalized = cp * scale + min_val
    return cp_denormalized


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


        #🎈🎈🎈🎈🎈🎈🎈🎈  TRAINING 🎈🎈🎈🎈🎈🎈🎈🎈
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


            # 🎈🎈🎈🎈🎈🎈🎈 FORWARD PASS 🎈🎈🎈🎈🎈🎈🎈
    
            x_normalized = normalize_input(x, scaler_features)
            cp = reg(x_normalized)
            cp_denormalized = denormalize_output(cp, scaler_target)
            #cp = reg(x) # prediction

            if method_name == "SPO+":
                # spo+ loss
                loss = func(cp_denormalized, c, w, z)
            elif  method_name == "2-Stage":
                loss = func(cp_denormalized, c)
            else:
                raise ValueError("Unknown method_name: {}".format(method_name))
            # regularization term
            #loss += 10 * l2(cp, c)


            # 🎈🎈🎈🎈🎈🎈🎈 BACKWARD PASS 🎈🎈🎈🎈🎈🎈🎈
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
num_epochs = 1   # number of epochs
lr = 1e-2         # learning rate
device = "cpu"    # device to use


from spoplus_locale import SPOPlus

method = 'SPO+'
func_spo = SPOPlus(model, processes=num_processes)

#func_2_stage = nn.MSELoss()

#method = "2-Stage"

print("Method:", method)
# training
logs = trainModel(reg, func_spo, method, loader_train, loader_test, model,
                    device=device, lr=lr, num_epochs=num_epochs)
# eval
regret_test = regret(reg, model, loader_test)


#🎈🎈🎈🎈🎈🎈🎈🎈 TEST 🎈🎈🎈🎈🎈🎈🎈🎈🎈

costo_totale = 0
# iterating over data loader
for i, data in enumerate(loader_test):
        # load data
        x, c, w, z = data
        # convert to numpy
        c = c.cpu().detach().numpy()[0]
        w = w.cpu().detach().numpy()[0]
        z = z.cpu().detach().numpy()[0]
        # predict with pytorch
        #x_normalized = normalize_input(x, scaler_features)
        
        cp = reg(x)

        #cp_denormalized = denormalize_output(cp, scaler_target)
        #cp_denormalized = cp_denormalized.cpu().detach().numpy()[0]
        model.setObj(cp)
        wp, _ = model.solve()
        costo_finestra = np.sum(-(cp * wp))
        #_, costo_finestra = model.solve()    #🎈🎈🎈🎈🎈🎈
        costo_totale += costo_finestra


print(f"COSTO TOTALE: {costo_totale:.2f} EURO")