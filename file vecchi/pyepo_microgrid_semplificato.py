from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, Param, NonNegativeReals, minimize, UnitInterval
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
from pyepo.model.omo import optOmoModel

from pyomo import environ as pe
from pyepo.model.omo import optOmoModel

from pyomo.environ import ConcreteModel, Var, ConstraintList, Objective, NonNegativeReals, Param, UnitInterval, minimize

from pyepo.model.opt import optModel


class myModel(optModel):
        def __init__(self, P_prod, P_load, solver='gurobi'):
            self.T = list(range(24))  # Orizzonte temporale di 24 ore
            self.P_prod = P_prod      # Produzione energetica
            self.P_load = P_load      # Consumo energetico

            # Parametri di esempio per la batteria
            self.P_B_cap = 200        # Capacità massima di energia stoccabile
            self.P_B_max_ch = 100     # Potenza massima di carica
            self.P_B_max_dis = 100    # Potenza massima di scarica
            self.ETA_B_ch = 0.95      # Efficienza di carica
            self.ETA_B_dis = 0.95     # Efficienza di scarica

            # Prezzi di acquisto e vendita dell'energia
            self.PR_buy = [0.15] * 24  # Costo per kWh acquistato
            self.PR_sell = [0.10] * 24 # Ricavo per kWh venduto

            self.solver = solver
            super().__init__()

        def _getModel(self):
            m = ConcreteModel()

            # Set del modello per l'orizzonte temporale
            m.T = self.T

            # Parametri
            m.P_prod = Param(m.T, initialize={t: self.P_prod[t] for t in m.T})
            m.P_load = Param(m.T, initialize={t: self.P_load[t] for t in m.T})
            m.PR_buy = Param(m.T, initialize={t: self.PR_buy[t] for t in m.T})
            m.PR_sell = Param(m.T, initialize={t: self.PR_sell[t] for t in m.T})

            # Variabili
            m.P_B_ch = Var(m.T, within=NonNegativeReals, bounds=(0, self.P_B_max_ch))
            m.P_B_dis = Var(m.T, within=NonNegativeReals, bounds=(0, self.P_B_max_dis))
            m.SOC_B = Var(m.T, within=UnitInterval)  # Stato di carica della batteria
            m.P_buy = Var(m.T, within=NonNegativeReals)
            m.P_sell = Var(m.T, within=NonNegativeReals)

            # Vincoli
            m.cons = ConstraintList()

            # Vincoli sullo stato di carica della batteria
            m.cons.add(m.SOC_B[0] == 0)  # SOC iniziale
            for t in m.T:
                if t > 0:
                    m.cons.add(m.SOC_B[t] == m.SOC_B[t-1] + (m.P_B_ch[t-1] * self.ETA_B_ch - m.P_B_dis[t-1] / self.ETA_B_dis) / self.P_B_cap)

            # Equilibrio dell'energia
            for t in m.T:
                m.cons.add(m.P_buy[t] - m.P_sell[t] + m.P_B_dis[t] - m.P_B_ch[t] == m.P_load[t] - m.P_prod[t])

            # Aggregazione delle variabili decisionali in un elenco o array
            x_vars = [m.P_B_ch, m.P_B_dis, m.SOC_B, m.P_buy, m.P_sell]

            return m, x_vars

        def setObj(self, c=None):
            model = self._model

            # Utilizza la funzione replace_objective per aggiornare l'obiettivo
            if hasattr(model, 'obj'):
                model.del_component('obj')

            # Funzione obiettivo
            model.obj = Objective(
                expr=sum(model.PR_buy[t] * model.P_buy[t] - model.PR_sell[t] * model.P_sell[t] for t in model.T),
                sense=minimize
            )

        def solve(self):
            solver = pe.SolverFactory(self.solver)
            result = solver.solve(self._model, tee=True)

            if result.solver.termination_condition == pe.TerminationCondition.optimal:
                # Ricava i valori delle variabili come array di float
                solution_list = []
                for var_object in self._model.component_objects(pe.Var, active=True):
                    for index in var_object:
                        val = var_object[index].value
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
                return None, None


if __name__ == '__main__':


    """## 3 Dataset and Data Loader"""


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

    pv_data = pd.read_csv('./Dataset/H2_W_resampled_15min.csv', usecols=[3])
    pv_data = pv_data.values[:35040].reshape(-1, 4).mean(axis=1)

    load_fix_data = pd.read_csv('./Dataset/Consumer_power.csv', usecols=[4])
    load_fix_data = load_fix_data.values[:35040].reshape(-1, 4).mean(axis=1)

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Dati storici (PV Data e Load Data)
    df = pd.DataFrame({
        'PV Data': pv_data,
        'Load Data': load_fix_data
    })

    # Creiamo le feature (x) e i costi obiettivo (c)
    """x = df[['PV Data', 'Load Data']].values
    c = -df['PV Data'].values + 0.2 * df['Load Data'].values  # Ad esempio, un costo approssimativo"""

    x = df[['PV Data', 'Load Data']].values[:100]
    c = (-df['PV Data'].values[:100] + 0.2 * df['Load Data'].values[:100]).reshape(-1, 1)  # Assicurati che sia un array 2D



    # Divisione in train e test
    x_train, x_test, c_train, c_test = train_test_split(x, c, test_size=0.2, random_state=42)

    import pyepo
    from pyepo.data.dataset import optDataset

    # Creiamo un'istanza del modello di ottimizzazione
    my_opt_model = myModel(pv_data, load_fix_data)

    # Generiamo il dataset di ottimizzazione
    train_dataset = optDataset(my_opt_model, x_train, c_train)
    test_dataset = optDataset(my_opt_model, x_test, c_test)


    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    # Definizione del modello predittivo
    class PredictiveModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(PredictiveModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            return self.linear(x)

    # Inizializziamo il modello
    input_dim = x_train.shape[1]  # Numero di feature
    output_dim = c_train.shape[1] if len(c_train.shape) > 1 else 1
    predictive_model = PredictiveModel(input_dim, output_dim)

    # Configuriamo i DataLoader per PyTorch
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Definiamo l'ottimizzatore
    optimizer = torch.optim.Adam(predictive_model.parameters(), lr=1e-3)

    from pyepo.func import SPOPlus

    # Inizializziamo la funzione di perdita SPO+
    spo_loss = SPOPlus(my_opt_model, processes=1)

    # Addestramento
    num_epochs = 2
    for epoch in range(num_epochs):
        for x_batch, c_true, w, z in train_loader:
            
            # Forward pass
            c_pred = predictive_model(x_batch)

            # Calcoliamo la perdita SPO+
            c_pred_expanded = c_pred.expand(-1, 120)
            c_true_expanded = c_true.expand(-1, 120)
            loss = spo_loss(c_pred_expanded, c_true_expanded, w, z)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")



