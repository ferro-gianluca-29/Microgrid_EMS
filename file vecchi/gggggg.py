# set random seed
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
import pandas as pd

# Carica i dati di produzione fotovoltaica
pv_data = pd.read_csv('./Dataset/H2_W_resampled_15min.csv', usecols=[3])
pv_data = pv_data.values[:35040].reshape(-1, 4).mean(axis=1)

# Carica i dati di consumo energetico
load_fix_data = pd.read_csv('./Dataset/Consumer_power.csv', usecols=[4])
load_fix_data = load_fix_data.values[:35040].reshape(-1, 4).mean(axis=1)

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

# Converti pv_data e load_fix_data in DataFrame
pv_df = pd.DataFrame(pv_data, columns=['pv_output'])
load_df = pd.DataFrame(load_fix_data, columns=['consumption'])

# Concatena i DataFrame (assumi che siano allineati temporalmente)
df = pd.concat([pv_df, load_df], axis=1)
lookback_window = 3  # giorni
hours_per_day = 24

# Assumi che ogni riga in `df` rappresenti un'ora
x = np.array([df.iloc[i * hours_per_day:(i + lookback_window) * hours_per_day].values for i in range(len(df) // hours_per_day - lookback_window)])
x = nanFill(x)  # Assicurati che la funzione nanFill gestisca correttamente la struttura dei dati
x = x.reshape(x.shape[0], -1)  # Appiattisci i dati di input

# Creiamo i target di previsione (c) basati sul consumo
c = np.array([df['consumption'].iloc[i * hours_per_day:(i + 1) * hours_per_day].values for i in range(lookback_window, len(df) // hours_per_day)])
c = nanFill(c)  # Riempi i possibili valori NaN

from sklearn.model_selection import train_test_split
x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=30, random_state=246)
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("c_train.shape:", c_train.shape)
print("c_test.shape:", c_test.shape)


from pyomo import environ as pe
from pyepo import EPO
from pyepo.model.omo import optOmoModel

class MicrogridModel(optOmoModel):
    def __init__(self, pv_data, load_data, solver='gurobi'):
        self.p_pv = pv_data
        self.p_demand = load_data
        super().__init__(solver=solver)

    def _getModel(self):
        m = pe.ConcreteModel()
        T = len(self.p_demand)  # Time periods
        
        # Decision variables
        m.p_gen = pe.Var(range(T), domain=pe.NonNegativeReals, bounds=lambda model, t: (0, self.p_pv[t]))  # PV generation within capacity
        m.p_use = pe.Var(range(T), domain=pe.NonNegativeReals)  # Energy usage
        
        # Objective function (for instance, minimizing energy not served)
        m.cost = pe.Objective(expr=sum((self.p_demand[t] - m.p_gen[t] - m.p_use[t])**2 for t in range(T)))
        
        # Constraints
        m.balance = pe.ConstraintList()
        for t in range(T):
            m.balance.add(m.p_gen[t] + m.p_use[t] >= self.p_demand[t])  # Energy balance constraint
        
        return m, m.p_gen
    
# Esempio di capacità e limiti
capacità_pv = df['pv_output'].max()  # Massima produzione PV osservata
capacità_accumulo = 100  # Capacità massima di scarica dell'accumulatore in kW

# Limite inferiore di produzione/scarica
plo_t = np.zeros(hours_per_day)  # Presumendo che non ci sia un minimo garantito

# Potenza programmata (basata su previsioni di consumo)
psch_t = df['consumption'].iloc[:hours_per_day].values  # Potresti voler adattare questo alle tue esigenze di previsione

# Limite superiore di produzione/scarica
pup_t = np.minimum(capacità_pv, capacità_accumulo)  # Il minore tra la capacità di PV e accumulo per ogni ora

optmodel = MicrogridModel(plo_t, psch_t, pup_t, solver = 'glpk')
optmodel.setObj(c_test[0])
sol, obj = optmodel.solve()
print('Obj: {}'.format(obj))
print('Sol: {}'.format(sol))

"""## 3 Dataset and Data Loader"""

from sklearn.preprocessing import MinMaxScaler

# Supponiamo che x_train e x_test siano già definiti come i tuoi dati di input
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

scaler2 = MinMaxScaler()
c_train = scaler2.fit_transform(c_train)
c_test = scaler2.transform(c_test)

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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
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
num_epochs = 2
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
