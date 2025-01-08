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
    def __init__(self, P_r_data, P_l_data,  T=24):
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
        m.P_r = Param(range(self.T), mutable=True, initialize={t: self.P_r_data[t] for t in range(self.T)})
        m.P_l = Param(range(self.T), mutable=True, initialize={t: self.P_l_data[t] for t in range(self.T)})

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


P_r = df['PV Data']
P_l = df['Load Data']
price = df['Price Data']

model = EnergyManagementSystem(P_r[:24], P_l[:24], T = 24) 

#🎈🎈🎈🎈🎈🎈 SOLUZIONE PER UNA FINESTRA 🎈🎈🎈🎈🎈🎈
# Imposta l'obiettivo di ottimizzazione
model.setObj(price[:24])
sol, obj = model.solve()

print(f"COSTO: {obj:.2f} EURO")

model.plot_variables()

# 🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈



