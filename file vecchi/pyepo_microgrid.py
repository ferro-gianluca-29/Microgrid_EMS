# set random seed
import random
random.seed(42)
import numpy as np
np.random.seed(42)
import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)
import pandas as pd


from sklearn.model_selection import train_test_split

from pyomo import environ as pe
from pyepo import EPO
from pyepo.model.omo import optOmoModel


class myModel(optOmoModel):

    def __init__(self):
        
        # Index
        self.I = list(range(3))  # CDG units
        self.T = list(range(24))  # time horizon

        # CDG
        self.C_CDG = [[140] * 24 for _ in self.I]  # simplified repeating costs
        self.C_SU = [2] * len(self.I)
        self.P_min = [0] * len(self.I)
        self.P_max = [500] * len(self.I)

        # Buying & selling prices
        self.PR_buy = [155] * len(self.T)
        self.PR_sell = [152.5] * len(self.T)

        self.P_wt = [0] * len(self.T)  # assuming zero wind

        self.vt_t = [[0] * len(self.T) for _ in self.T]  # penalty of load shifting
        self.IF_max = [0] * 8 + [90, 35] + [0] * 3 + [105, 41, 36, 51, 59, 45, 13, 0, 8] + [0] * 5
        self.OF_max = [0] * 8 + [410, 465, 500, 395, 459, 464, 449, 441, 455, 487, 500, 492] + [0] * 4

        # Battery specs
        self.L_B_ch = 0.03
        self.L_B_dis = 0.03
        self.P_B_cap = 200
        self.P_BTB = 200
        self.ETA_BTB = 0.98
        self.DELTA_B = 0.03

        self.M = 1e9

        super().__init__(solver='gurobi')
        

    def _getModel(self):


        m = pe.ConcreteModel()

        self._model = m

        # Sets
        m.ZeroOrOne = pe.Set(initialize=[0, 1])

        # Parameters
        m.M = pe.Param(initialize=self.M)

        # CDG Variables
        m.P_CDG = pe.Var(self.I, self.T, within=pe.NonNegativeReals)
        m.y = pe.Var(self.I, self.T, within=m.ZeroOrOne)
        m.u = pe.Var(self.I, self.T, within=m.ZeroOrOne, initialize=0)
        m.i_1 = pe.Var(self.I, self.T, within=m.ZeroOrOne)
        m.i_2 = pe.Var(self.I, self.T, within=m.ZeroOrOne)

        # Battery Variables
        m.P_B_ch = pe.Var(self.T, within=pe.NonNegativeReals)
        m.P_B_dis = pe.Var(self.T, within=pe.NonNegativeReals)
        m.SOC_Bp = pe.Var(self.T, within=pe.UnitInterval)
        m.SOC_B = pe.Var(self.T, within=pe.UnitInterval)

        # Load Variables
        m.P_L_adj = pe.Var(self.T, within=pe.NonNegativeReals)
        m.P_sh = pe.Var(self.T, self.T, within=pe.NonNegativeReals)

        # Transaction Variables
        m.P_short = pe.Var(self.T, within=pe.NonNegativeReals, initialize=0)
        m.P_sur = pe.Var(self.T, within=pe.NonNegativeReals, initialize=0)

        # Constraints
        m.cons = pe.ConstraintList()

        # CDG Power Limits
        for i in self.I:
            for t in self.T:
                m.cons.add(m.P_CDG[i, t] >= m.u[i, t] * self.P_min[i])
                m.cons.add(m.P_CDG[i, t] <= m.u[i, t] * self.P_max[i])

        # Y Value Constraints
        for i in self.I:
            for t in self.T:
                if t == 0:
                    m.cons.add(m.y[i, t] >= m.u[i, t] - 0)  # Assuming initial u0 is 0
                else:
                    m.cons.add(m.y[i, t] >= m.u[i, t] - m.u[i, t-1])

        # Power Balance
        for t in self.T:
            m.cons.add(
                sum(m.P_CDG[i, t] for i in self.I) + m.P_short[t] + m.P_B_dis[t] +
                sum(m.P_sh[tp, t] for tp in self.T if tp != t) ==  # Assicurati che l'inflow sia considerato
                m.P_L_adj[t] + m.P_sur[t] + m.P_B_ch[t] +
                sum(m.P_sh[t, tp] for tp in self.T if tp != t)  # Assicurati che l'outflow sia considerato
            )

        # Battery Constraints
        for t in self.T:
            if t == 0:
                m.cons.add(m.P_B_ch[t] <= self.P_B_cap * (1 - 0) / (1 - self.L_B_ch) / self.ETA_BTB)  # Assuming initial SOC_B0 is 0
                m.cons.add(m.P_B_dis[t] <= self.P_B_cap * 0 * (1 - self.L_B_dis) * self.ETA_BTB)  # Assuming initial SOC_B0 is 0
            else:
                m.cons.add(m.P_B_ch[t] <= self.P_B_cap * (1 - m.SOC_B[t-1]) / (1 - self.L_B_ch) / self.ETA_BTB)
                m.cons.add(m.P_B_dis[t] <= self.P_B_cap * m.SOC_B[t-1] * (1 - self.L_B_dis) * self.ETA_BTB)

        # SOC Update and Self-discharge
        for t in self.T:
            if t == 0:
                m.cons.add(m.SOC_B[t] == 0 - (1 / self.P_B_cap) * ((1 / (1 - self.L_B_dis) / self.ETA_BTB * m.P_B_dis[t]) - (m.P_B_ch[t] * (1 - self.L_B_ch) * self.ETA_BTB)))
            else:
                m.cons.add(m.SOC_B[t] == m.SOC_B[t-1] - (1 / self.P_B_cap) * ((1 / (1 - self.L_B_dis) / self.ETA_BTB * m.P_B_dis[t]) - (m.P_B_ch[t] * (1 - self.L_B_ch) * self.ETA_BTB)))
            m.cons.add(m.SOC_B[t] == (1 - self.DELTA_B) * m.SOC_Bp[t])

        return m, None 

    def setObj(self, c=None):
        model = self._model

        # Rimuovi l'obiettivo esistente, se presente
        if hasattr(model, 'obj'):
            model.del_component('obj')

        if c is None:
            c = [[140] * 24 for _ in self.I]  # Usa il costo di default se c non è fornito

        # Se c è un vettore, convertilo in una matrice replicata per ogni unità CDG
        if isinstance(c, np.ndarray) and c.ndim == 1:
            c = np.tile(c, (len(self.I), 1)).tolist()  # Converti in lista per Pyomo

        # Assicurati che c sia ora in formato matrice
        if isinstance(c, list) and all(len(row) == len(self.T) for row in c):
            cdg_cost = sum(c[i][t] * model.P_CDG[i, t] + self.C_SU[i] * model.y[i, t] for i in self.I for t in self.T)
        else:
            raise ValueError("Cost parameter 'c' is not a properly structured matrix.")

        transaction_cost = sum(self.PR_buy[t] * model.P_short[t] - self.PR_sell[t] * model.P_sur[t] for t in self.T)
        load_shift_penalty = sum(self.vt_t[t][tp] * model.P_sh[t, tp] for t in self.T for tp in self.T if t != tp)

        # Imposta il costo totale come obiettivo da minimizzare
        model.obj = pe.Objective(expr=cdg_cost + transaction_cost + load_shift_penalty, sense=pe.minimize)


    def solve(self):
        solver = pe.SolverFactory(self.solver)
        result = solver.solve(self._model, tee=True)

        if result.solver.termination_condition == pe.TerminationCondition.optimal:
            # Ricava i valori delle variabili come array di float
            solution_list = []
            for var_object in self._model.component_objects(pe.Var, active=True):
                for index in var_object:
                    val = var_object[index].value
                    # Controlla eventuali None o valori non numerici
                    if val is None:
                        val = 0.0  # oppure un'altra gestione a tua scelta
                    solution_list.append(float(val))
            
            solution_array = np.array(solution_list, dtype=float)
            objective_value = float(pe.value(self._model.obj))
            if not np.all(np.isfinite(solution_array)):
                print("Attenzione: trovati NaN/inf nella soluzione, li converto a 0.")
            solution_array = np.nan_to_num(solution_array, nan=0.0, posinf=1e6, neginf=-1e6)
            return solution_array, objective_value
        else:
            # gestione del caso non ottimale
            return None, None

        
"""## 3 Dataset and Data Loader"""

pv_data = pd.read_csv('./Dataset/H2_W_resampled_15min.csv', usecols=[3])
pv_data = pv_data.values[:35040].reshape(-1, 4).mean(axis=1)

load_fix_data = pd.read_csv('./Dataset/Consumer_power.csv', usecols=[4])
load_fix_data = load_fix_data.values[:35040].reshape(-1, 4).mean(axis=1)

df = pd.DataFrame({
    'PV Data': pv_data,
    'Load Data': load_fix_data
})

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

lookback_window = 3
x = np.array([df.iloc[d * 24:(d + lookback_window) * 24].values.flatten() for d in range(0, 365 - lookback_window)])
c = np.array([df.iloc[d * 24:(d + 1) * 24, 0].values for d in range(lookback_window, 365)])

x = nanFill(x)
c = nanFill(c)

# Dividi i dati in training e test
from sklearn.model_selection import train_test_split
x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=30, random_state=246)

# Pulizia e conversione dei dati di costo
c_train = np.nan_to_num(c_train).astype(float)
c_test = np.nan_to_num(c_test).astype(float)

# Assicurati che anche le features siano pulite e nel formato corretto
x_train = np.nan_to_num(x_train).astype(float)
x_test = np.nan_to_num(x_test).astype(float)



# Verifica delle dimensioni
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("c_train.shape:", c_train.shape)
print("c_test.shape:", c_test.shape)

optmodel = myModel()
optmodel.setObj(c_test[0])
sol, obj = optmodel.solve()
print('Obj: {}'.format(obj))
print('Sol: {}'.format(sol))

import pyepo
dataset_train = pyepo.data.dataset.optDataset(optmodel, x_train, c_train)
dataset_test = pyepo.data.dataset.optDataset(optmodel, x_test, c_test)

from torch.utils.data import DataLoader
batch_size = 4
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)


