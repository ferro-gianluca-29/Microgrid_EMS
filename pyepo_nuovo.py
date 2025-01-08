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
# convert the datetime format
df_load.index = pd.to_datetime(df_load.index, format="%d.%m.%Y %H:%M")
# remove the index name
df_load.index.name = None
# resample to 1h resolution
df_load = df_load.resample("1H").mean()
# rename the column
df_load = df_load.rename('load')
# to dataframe
df_load = pd.DataFrame(df_load)
# display
print(df_load.head())


from datetime import datetime, timedelta
# base date to start data
base = datetime(2022, 1, 1)
# number of days to import starting from base
date_list = [base + timedelta(days=x) for x in range(366)]



df_price = pd.read_csv("./price_data.csv")

# display
print(df_price.head())


df_climate = pd.read_csv('climate_data.csv', index_col=0, parse_dates=True)


# random flexibility bound of demand
plo_t = df_load.load.values * (0.7 + 0.1 * np.random.rand(df_load.values.size)) # lower bound
pup_t = df_load.load.values * (1.2 + 0.1 * np.random.rand(df_load.values.size)) # upper bound
# joint dataset
df = pd.DataFrame({"pi_t": df_price.price.values,                               # price
                   "temp": df_climate.temp.values[:df_price.shape[0]],          # temperature
                   "rhum": df_climate.rhum.values[:df_price.shape[0]],          # relative humidity
                   "wdir": df_climate.wdir.values[:df_price.shape[0]],          # wind direction
                   "wspd": df_climate.wspd.values[:df_price.shape[0]],          # wind speed
                   "pres": df_climate.pres.values[:df_price.shape[0]],          # pressure
                   "tsun": df_climate.tsun.values[:df_price.shape[0]],          # number of minutes of sunshine in the hour
                   "Psch_t": df_load.load.values,                               # scheduled load
                   "Plo_t": plo_t,                                              # flexibility lower bound of load
                   "Pup_t": pup_t                                               # flexibility upper bound of load
    })
print(df.head())

def nanFill(x):
    """
    A function to replaces NaN values with the mean of the previous and next values.
    """
    if x.ndim == 2:
        # iterate over columns in a 2D array
        for j in range(x.shape[1]):
            # get index of nan
            nan_indices = np.where(np.isnan(x[:, j]))[0]
            for i in nan_indices:
                # fill nan by mean
                if 0 < i < len(x) - 1:
                    x[i, j] = np.nanmean([x[i - 1, j], x[i + 1, j]])
                # boundary cases
                elif i == 0:
                    x[i, j] = x[i + 1, j]
                elif i == len(x) - 1:
                    x[i, j] = x[i - 1, j]
    elif x.ndim == 3:
        # iterate over 3rd dimension in a 3D array
        for k in range(x.shape[2]):
            for j in range(x.shape[1]):
                # get index of nan
                nan_indices = np.where(np.isnan(x[:, j, k]))[0]
                for i in nan_indices:
                    # fill nan by mean
                    if 0 < i < len(x) - 1:
                        x[i, j, k] = np.nanmean([x[i - 1, j, k], x[i + 1, j, k]])
                    # boundary cases
                    elif i == 0:
                        x[i, j, k] = x[i + 1, j, k]
                    elif i == len(x) - 1:
                        x[i, j, k] = x[i - 1, j, k]
    return x

# get features
df_feat = df[["pi_t", "temp", "wdir"]].reset_index(drop=True)
# features and costs
lookback_window = 3
# observable features (price, temperature, and wind direction for previous day)
x = np.array([df_feat.iloc[d * 24:(d + lookback_window) * 24, :].values for d in range(0, 365 - lookback_window)])
x = nanFill(x)
# reshape the 3D arrays to 2D arrays
x = x.reshape(x.shape[0], -1)
# labeled costs (true electricity price vector for tomorrow)
c = np.array([df.iloc[d * 24:(d + 1) * 24, 1].values for d in range(lookback_window, 365)])
c = nanFill(c)
# get the flexibility bounds and scheduled load
plo_t = df.loc[:23, 'Plo_t'].values
psch_t = df.loc[:23, 'Psch_t'].values
pup_t = df.loc[:23, 'Pup_t'].values

# data split
from sklearn.model_selection import train_test_split
x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=30, shuffle=False, random_state=246)
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("c_train.shape:", c_train.shape)
print("c_test.shape:", c_test.shape)


import gurobipy as gp
from pyepo.model.grb import optGrbModel

class DemandResponse(optGrbModel):
    """
    This class is an optimization model for energy scheduling.

    Attributes:
        _model (model): a Pyomo model
        plo_t (np.ndarray / list): lower bound of the demand
        psch_t (np.ndarray / list): scheduled  demand
        pup_t (np.ndarray / list): upper bound of the demand
    """

    def __init__(self, plo_t, psch_t, pup_t):
        """
        Args:
            plo_t (np.ndarray / list): lower bound of the demand
            psch_t (np.ndarray / list): scheduled  demand
            pup_t (np.ndarray / list): upper bound of the demand
        """
        self.plo_t = np.array(plo_t)
        self.psch_t = np.array(psch_t)
        self.pup_t = np.array(pup_t)
        super().__init__()

    def _getModel(self):
        # create a model
        m = gp.Model("Energy")
        # variables
        x = m.addMVar(24, lb=self.plo_t, ub=self.pup_t, name="x")
        # constr
        m.addConstr(gp.quicksum(x) == gp.quicksum(self.psch_t))
        return m, x

# init model
optmodel = DemandResponse(plo_t, psch_t, pup_t)


# solve model (just test)
optmodel.setObj(c_test[0]) # set objective function 
sol, obj = optmodel.solve() # solve
# print res
print('Obj: {}'.format(obj))
print('Sol: {}'.format(sol))

"""## 3 Dataset and Data Loader

> "PyTorch provides two data primitives: ``Dataset`` and ``DataLoader`` that allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples. "  -- PyTorch Documentation

``optDataset`` is extended from PyTorch ``Dataset``. In order to obtain optimal solutions, ``optDataset`` requires the corresponding ``optModel`` is a module of PyEPO library, which is designed as a container for any "black box" solver. The tutorial on ``optModel`` is [here](https://github.com/khalil-research/PyEPO/blob/main/notebooks/01%20Optimization%20Model.ipynb).
"""

import pyepo

from pyepo.data.dataset import optDataset

# get training data set
dataset_train = optDataset(optmodel, x_train, c_train)
# get test data set
dataset_test = optDataset(optmodel, x_test, c_test)

# get data loader
from torch.utils.data import DataLoader
batch_size = 4
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

"""## 4 PyTorch Linear Regression

PyTorch is an open-source machine learning library primarily used for developing and training deep learning models such as neural networks. It is developed by Facebook's AI Research lab and is based on the Torch library. PyTorch provides a flexible and intuitive interface for creating and training models.

In PyTorch, the ``nn.Module`` is a base class for all neural network modules in the library. It provides a convenient way to organize the layers of a model, and to define the forward pass of the model.

Here, we build a MLP with a hidden layer.
"""

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
input_dim = x.shape[-1]
output_dim = c.shape[-1]
hidden_dim = 48
# init for test
reg = multiLayerPerceptron(input_dim, hidden_dim, output_dim)

"""## 5 Training and Testing

**The** core capability of PyEPO is to embed the optimization model into an artificial neural network for the end-to-end training. For this purpose, PyEPO includes several different methods.

### 5.1 Visualization Functions
"""

import matplotlib.pyplot as plt
from matplotlib import cm

def visSol(plo_t, pup_t, data_loader, optmodel, ind=0,
           pytorch_model=None, sklearn_model=None, method_name=None):
    # iterating over data loader
    for i, data in enumerate(data_loader):
        if i == ind:
            # load data
            x, c, w, z = data
            # move to GPU if available
            if torch.cuda.is_available():
                x = x.cuda()
            # convert to numpy
            c = c.cpu().detach().numpy()[0]
            w = w.cpu().detach().numpy()[0]
            z = z.cpu().detach().numpy()[0]
            # predict with pytorch
            if pytorch_model is not None:
                cp = pytorch_model(x)
                cp = cp.cpu().detach().numpy()[0]
            # predict with sklearn
            elif sklearn_model is not None:
                x = x.cpu().detach().numpy()
                cp = sklearn_model.predict(x)[0]
            # ground truth
            else:
                cp = c
            # plot
            fig = plotSol(c, cp, plo_t, pup_t, ind, method_name)
            break

# plot function
def plotSol(c, cp, plo_t, pup_t, ind, method):
    # solve get optimal solution
    optmodel.setObj(c)
    w, _ = optmodel.solve()
    optmodel.setObj(cp)
    wp, _ = optmodel.solve()
    # calculate total cost
    total_cost = np.sum(c * wp)
    # hours
    t = range(24)
    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    # figure 1: price
    axs[0].plot(t, cp, color="b", linewidth=2, linestyle="--", label="Predicted Price")
    axs[0].plot(t, c, color="orange", linewidth=2, label="True Price")
    axs[0].legend(loc="upper left", shadow=False)
    axs[0].set_xlabel("Time (Hour)", fontsize=12)
    axs[0].set_ylabel("Price (€/kWh)", fontsize=12)
    axs[0].set_title("Price Prediction")
    # figure 2: load
    axs[1].plot(t, plo_t, color="r", linewidth=1, alpha=0.8, label="Bounds")
    axs[1].plot(t, pup_t, color="r", alpha=0.8, linewidth=1)
    axs[1].plot(t, wp, color="b", linewidth=2, linestyle="--", marker=".", label="Predicted Load", zorder=4)
    axs[1].plot(t, w, color="orange", linewidth=2, marker=".", label="True Load")
    axs[1].legend(loc="upper left", shadow=False)
    axs[1].set_xlabel("Time (Hour)", fontsize=12)
    axs[1].set_ylabel("Demand (p.u.)", fontsize=12)
    axs[1].set_title("Load Decision")
    # add title
    fig.suptitle("Instance {}, Total Cost for {}: {:.2f}".format(ind, method, total_cost))
    plt.show()

def plotLearningCurve(logs, method):
    # plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=False)
    # figure 1: loss
    axs[0].plot(logs["loss"], color="c", linewidth=1)
    # set the x and y axis tick parameters
    axs[0].tick_params(axis="x", labelsize=10)
    axs[0].tick_params(axis="y", labelsize=10)
    # set the x axis limits
    axs[0].set_xlim(-1, len(logs["loss"])+1)
    # set the labels and title
    axs[0].set_xlabel("Iters", fontsize=12)
    axs[0].set_ylabel("Loss", fontsize=12)
    axs[0].set_title("Loss Curve on Training Set")
    # figure 2: regret
    axs[1].plot(100*logs["regret_train"], color="c", linewidth=2, label="Training")
    axs[1].plot(100*logs["regret_test"], color="g", linewidth=2, label="Test")
    # set the x and y axis tick parameters
    axs[1].tick_params(axis="x", labelsize=10)
    axs[1].tick_params(axis="y", labelsize=10)
    # set the x axis limits
    axs[1].set_xlim(-0.2, len(logs["regret_train"])-1+0.2)
    axs[1].set_ylim(0, 5)
    # set the labels and title
    axs[1].set_xlabel("Epochs", fontsize=12)
    axs[1].set_ylabel("Regret (%)", fontsize=12)
    axs[1].legend(loc="upper right", shadow=False)
    axs[1].set_title("Regret Curve on Training and Test Set")
    # add title
    fig.suptitle("Learning Curve for {}, Training Time: {:.2f} sec".format(method, logs["elapsed"]))
    plt.show()

"""### 5.2 Training"""

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
            # forward pass
            cp = reg(x) # prediction
            if method_name == "SPO+":
                # spo+ loss
                loss = func(cp, c, w, z)
            else:
                raise ValueError("Unknown method_name: {}".format(method_name))
            # regularization term
            #loss += 10 * l2(cp, c)
            # backward pass
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
num_epochs = 5    # number of epochs
lr = 1e-2         # learning rate
device = "cpu"    # device to use

# autograd functions
func_dict = {
    "2-Stage": nn.MSELoss(),
    "SPO+" :pyepo.func.SPOPlus(optmodel, processes=num_processes),
    "DBB": pyepo.func.blackboxOpt(optmodel, lambd=20, processes=num_processes),
    "NID": pyepo.func.negativeIdentity(optmodel, processes=num_processes),
    "DPO": pyepo.func.perturbedOpt(optmodel, n_samples=1, sigma=0.5, processes=num_processes),
    "PFYL": pyepo.func.perturbedFenchelYoung(optmodel, n_samples=1, sigma=0.5, processes=num_processes),
    "I-MLE": pyepo.func.implicitMLE(optmodel, n_samples=1, sigma=0.5, lambd=20, processes=num_processes),
    "AI-MLE": pyepo.func.adaptiveImplicitMLE(optmodel, n_samples=1, sigma=0.5, processes=num_processes),
    "NCE": pyepo.func.NCE(optmodel, processes=num_processes, solve_ratio=0.05, dataset=dataset_train),
    "Listwise LTR": pyepo.func.listwiseLTR(optmodel, processes=num_processes, solve_ratio=0.05, dataset=dataset_train),
    "Pairwise LTR": pyepo.func.pairwiseLTR(optmodel, processes=num_processes, solve_ratio=0.05, dataset=dataset_train),
    "Pointwise LTR": pyepo.func.pointwiseLTR(optmodel, processes=num_processes, solve_ratio=0.05, dataset=dataset_train)
}


from pyepo.func import SPOPlus

method = 'SPO+'
func = SPOPlus(optmodel, processes=num_processes)

print("Method:", method)
# init model
# reg = LinearRegressionNN()
reg = multiLayerPerceptron(input_dim, hidden_dim, output_dim)
# training
logs = trainModel(reg, func, method, loader_train, loader_test, optmodel,
                    device=device, lr=lr, num_epochs=num_epochs)
# eval
regret_test = regret(reg, optmodel, loader_test)

# draw plot
plotLearningCurve(logs, method)
visSol(plo_t, pup_t, loader_test, optmodel, ind=10, pytorch_model=reg, method_name=method)



from pyepo.model.grb  import knapsackModel