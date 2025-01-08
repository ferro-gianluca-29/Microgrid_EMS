# set random seed
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
import pandas as pd

# load the dataset from a CSV file
df = pd.read_csv('./LoadProfile.csv',sep=';')
print('The size of the dataset is ({:},{:})'.format(len(df.index),len(df.columns)))
print(df.head())

# set 'time' as the index and select the 'mv_comm_pload' column
df_load = df.set_index("time").loc[:,"mv_comm_pload"]
df_load.index = pd.to_datetime(df_load.index, format="%d.%m.%Y %H:%M")
df_load.index.name = None
df_load = df_load.resample("1H").mean()
df_load = df_load.rename('load')
df_load = pd.DataFrame(df_load)
print(df_load.head())

"""from datetime import datetime, timedelta
base = datetime(2022, 1, 1)
date_list = [base + timedelta(days=x) for x in range(366)]"""

df_price = pd.read_csv("./price_data.csv")
print(df_price.head())

df_climate = pd.read_csv('climate_data.csv', index_col=0, parse_dates=True)

plo_t = df_load.load.values * (0.7 + 0.1 * np.random.rand(df_load.values.size))
pup_t = df_load.load.values * (1.2 + 0.1 * np.random.rand(df_load.values.size))

df = pd.DataFrame({"pi_t": df_price.price.values,
                   "temp": df_climate.temp.values[:df_price.shape[0]],
                   "rhum": df_climate.rhum.values[:df_price.shape[0]],
                   "wdir": df_climate.wdir.values[:df_price.shape[0]],
                   "wspd": df_climate.wspd.values[:df_price.shape[0]],
                   "pres": df_climate.pres.values[:df_price.shape[0]],
                   "tsun": df_climate.tsun.values[:df_price.shape[0]],
                   "Psch_t": df_load.load.values,
                   "Plo_t": plo_t,
                   "Pup_t": pup_t
    })
print(df.head())

def nanFill(x):
    if x.ndim == 2:
        for j in range(x.shape[1]):
            nan_indices = np.where(np.isnan(x[:, j]))[0]
            for i in nan_indices:
                if 0 < i < len(x) - 1:
                    x[i, j] = np.nanmean([x[i - 1, j], x[i + 1, j]])
                elif i == 0:
                    x[i, j] = x[i + 1, j]
                elif i == len(x) - 1:
                    x[i, j] = x[i - 1, j]
    elif x.ndim == 3:
        for k in range(x.shape[2]):
            for j in range(x.shape[1]):
                nan_indices = np.where(np.isnan(x[:, j, k]))[0]
                for i in nan_indices:
                    if 0 < i < len(x) - 1:
                        x[i, j, k] = np.nanmean([x[i - 1, j, k], x[i + 1, j, k]])
                    elif i == 0:
                        x[i, j, k] = x[i + 1, j, k]
                    elif i == len(x) - 1:
                        x[i, j, k] = x[i - 1, j, k]
    return x

df_feat = df[["pi_t", "temp", "wdir"]].reset_index(drop=True)
lookback_window = 3
x = np.array([df_feat.iloc[d * 24:(d + lookback_window) * 24, :].values for d in range(0, 365 - lookback_window)])
x = nanFill(x)
c = np.array([df.iloc[d * 24:(d + 1) * 24, 0].values for d in range(lookback_window, 365)])
c = nanFill(c)
plo_t = df.loc[:23, 'Plo_t'].values
psch_t = df.loc[:23, 'Psch_t'].values
pup_t = df.loc[:23, 'Pup_t'].values


from sklearn.model_selection import train_test_split
x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=30, shuffle=False, random_state=246)
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("c_train.shape:", c_train.shape)
print("c_test.shape:", c_test.shape)


from pyomo import environ as pe
from pyepo import EPO
from pyepo.model.omo import optOmoModel

class myModel(optOmoModel):

    def __init__(self, plo_t, psch_t, pup_t, solver='gurobi'):
        self.plo_t = np.array(plo_t)
        self.psch_t = np.array(psch_t)
        self.pup_t = np.array(pup_t)
        super().__init__(solver=solver)  # Passa il parametro solver al costruttore della classe padre

    def _getModel(self):
        m = pe.ConcreteModel()

        # Funzione per definire i limiti delle variabili
        def bounds_func(model, i):
            return (self.plo_t[i], self.pup_t[i])

        # Definizione delle variabili con i rispettivi limiti
        m.x = pe.Var(range(24), domain=pe.Reals, bounds=bounds_func)

        # Definizione del vincolo di somma
        m.cons = pe.Constraint(expr=sum(m.x[i] for i in range(24)) == sum(self.psch_t))

        # Assegnazione del modello e delle variabili all'oggetto
        self._model = m
        self.x = m.x

        # Modifica qui per restituire sia il modello che le variabili come una tupla
        return m, m.x  # Restituisce sia il modello che le variabili

optmodel = myModel(plo_t, psch_t, pup_t, solver = 'glpk')
optmodel.setObj(c_test[0])
sol, obj = optmodel.solve()
print('Obj: {}'.format(obj))
print('Sol: {}'.format(sol))

"""## 3 Dataset and Data Loader"""

def calculate_scale_params(data):
    min_vals = data.min(axis=(0, 1), keepdims=True)
    max_vals = data.max(axis=(0, 1), keepdims=True)
    return min_vals, max_vals

def apply_normalization(data, min_vals, max_vals):
    return (data - min_vals) / (max_vals - min_vals)

# Calcola i parametri di scala dai dati di training
min_vals_x, max_vals_x = calculate_scale_params(x_train)
min_vals_c, max_vals_c = calculate_scale_params(c_train)

# Applica la normalizzazione ai dati di training
x_train = apply_normalization(x_train, min_vals_x, max_vals_x)
c_train = apply_normalization(c_train, min_vals_c, max_vals_c)

# Applica la normalizzazione ai dati di test usando i parametri di training
x_test = apply_normalization(x_test, min_vals_x, max_vals_x)
c_test = apply_normalization(c_test, min_vals_c, max_vals_c)

import pyepo
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)


from torch.utils.data import DataLoader
batch_size = 4
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)


## LSTM Regressor"""

import torch
from torch import nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMRegressor, self).__init__()
        # Definizione degli attributi basati sui parametri passati al costruttore
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Inizializzazione del modulo LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Inizializzazione del layer fully-connected
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Inizializzazione degli stati nascosti h0 e c0 a zero
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        # Applicazione del modulo LSTM
        out, _ = self.lstm(x, (h0, c0))
        # Applicazione del layer fully-connected all'ultimo output della sequenza
        out = self.fc(out[:, -1, :])
        return out

input_dim = x.shape[-1]
output_dim = c.shape[-1]
hidden_dim = 48
reg = LSTMRegressor(input_dim, hidden_dim, output_dim)

## 5 Training and Testing

import matplotlib.pyplot as plt
from matplotlib import cm

def visSol(plo_t, pup_t, data_loader, optmodel, ind=0,
           pytorch_model=None, sklearn_model=None, method_name=None):
    for i, data in enumerate(data_loader):
        if i == ind:
            x, c, w, z = data
            if torch.cuda.is_available():
                x = x.cuda()
            c = c.cpu().detach().numpy()[0]
            w = w.cpu().detach().numpy()[0]
            z = z.cpu().detach().numpy()[0]
            if pytorch_model is not None:
                cp = pytorch_model(x)
                cp = cp.cpu().detach().numpy()[0]
            elif sklearn_model is not None:
                x = x.cpu().detach().numpy()
                cp = sklearn_model.predict(x)[0]
            else:
                cp = c
            fig = plotSol(c, cp, plo_t, pup_t, ind, method_name)
            break

def plotSol(c, cp, plo_t, pup_t, ind, method):
    optmodel.setObj(c)
    w, _ = optmodel.solve()
    optmodel.setObj(cp)
    wp, _ = optmodel.solve()
    total_cost = np.sum(c * wp)
    t = range(24)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axs[0].plot(t, cp, color="b", linewidth=2, linestyle="--", label="Predicted Price")
    axs[0].plot(t, c, color="orange", linewidth=2, label="True Price")
    axs[0].legend(loc="upper left", shadow=False)
    axs[0].set_xlabel("Time (Hour)", fontsize=12)
    axs[0].set_ylabel("Price (€/kWh)", fontsize=12)
    axs[0].set_title("Price Prediction")
    axs[1].plot(t, plo_t, color="r", linewidth=1, alpha=0.8, label="Bounds")
    axs[1].plot(t, pup_t, color="r", alpha=0.8, linewidth=1)
    axs[1].plot(t, wp, color="b", linewidth=2, linestyle="--", marker=".", label="Predicted Load", zorder=4)
    axs[1].plot(t, w, color="orange", linewidth=2, marker=".", label="True Load")
    axs[1].legend(loc="upper left", shadow=False)
    axs[1].set_xlabel("Time (Hour)", fontsize=12)
    axs[1].set_ylabel("Demand (p.u.)", fontsize=12)
    axs[1].set_title("Load Decision")
    fig.suptitle("Instance {}, Total Cost for {}: {:.2f}".format(ind, method, total_cost))
    plt.show()

def plotLearningCurve(logs, method):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    axs[0].plot(logs["loss"], color="c", linewidth=1)
    axs[0].tick_params(axis="x", labelsize=10)
    axs[0].tick_params(axis="y", labelsize=10)
    axs[0].set_xlim(-1, len(logs["loss"])+1)
    axs[0].set_xlabel("Iters", fontsize=12)
    axs[0].set_ylabel("Loss", fontsize=12)
    axs[0].set_title("Loss Curve on Training Set")
    axs[1].plot(100*logs["regret_train"], color="c", linewidth=2, label="Training")
    axs[1].plot(100*logs["regret_test"], color="g", linewidth=2, label="Test")
    axs[1].tick_params(axis="x", labelsize=10)
    axs[1].tick_params(axis="y", labelsize=10)
    axs[1].set_xlim(-0.2, len(logs["regret_train"])-1+0.2)
    axs[1].set_ylim(0, 5)
    axs[1].set_xlabel("Epochs", fontsize=12)
    axs[1].set_ylabel("Regret (%)", fontsize=12)
    axs[1].legend(loc="upper right", shadow=False)
    axs[1].set_title("Regret Curve on Training and Test Set")
    fig.suptitle("Learning Curve for {}, Training Time: {:.2f} sec".format(method, logs["elapsed"]))
    plt.show()

import time
from tqdm import tqdm

def trainModel(reg, func, method_name, loader_train, loader_test, optmodel,
                device="cpu", lr=1e-1, num_epochs=1):
    optimizer = torch.optim.Adam(reg.parameters(), lr=lr)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    reg.train()
    loss_log, regret_log_train, regret_log_test = [], [], []
    tbar = tqdm(range(num_epochs))
    elapsed = 0
    for epoch in tbar:
        regret_train = pyepo.metric.regret(reg, optmodel, loader_train)
        regret_log_train.append(regret_train)
        regret_test = pyepo.metric.regret(reg, optmodel, loader_test)
        regret_log_test.append(regret_test)
        tick = time.time()
        for i, data in enumerate(loader_train):
            x, c, w, z = data
            x, c, w, z = x.to(device), c.to(device), w.to(device), z.to(device)
            cp = reg(x)
            if method_name == "SPO+":
                loss = func(cp, c, w, z)
            # altri metodi eliminati
            else:
                raise ValueError("Unknown method_name: {}".format(method_name))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_log.append(loss.item())
            tbar.set_description("Epoch: {:2}, Loss: {:3.4f}".format(epoch, loss.item()))
            tock = time.time()
            elapsed += tock - tick
    regret_train = pyepo.metric.regret(reg, optmodel, loader_train)
    regret_log_train.append(regret_train)
    regret_test = pyepo.metric.regret(reg, optmodel, loader_test)
    regret_log_test.append(regret_test)
    logs = {}
    logs["loss"] = np.array(loss_log)
    logs["regret_train"] = np.array(regret_log_train)
    logs["regret_test"] = np.array(regret_log_test)
    logs["elapsed"] = elapsed
    print("{}: Regret on test set: {:.2f}% after {:.2f} sec of training".format(method_name, regret_test*100, elapsed))
    return logs

num_processes = 1
num_epochs = 5
lr = 1e-2
device = "cpu"

method = 'SPO+'
func = pyepo.func.SPOPlus(optmodel, processes=num_processes)

print("Method:", method)
reg = LSTMRegressor(input_dim, hidden_dim, output_dim)
logs = trainModel(reg, func, method, loader_train, loader_test, optmodel,
                    device=device, lr=lr, num_epochs=num_epochs)
regret_test = pyepo.metric.regret(reg, optmodel, loader_test)
plotLearningCurve(logs, method)
visSol(plo_t, pup_t, loader_test, optmodel, ind=0, pytorch_model=reg, method_name=method)
