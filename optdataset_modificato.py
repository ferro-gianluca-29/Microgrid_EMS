import time
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from pyepo.model.opt import optModel

class optDataset(Dataset):
    """
    This class is a Torch Dataset for optimization problems that dynamically updates
    parameters related to renewable generation and load profiles.

    Attributes:
        model (optModel): Optimization models
        feats (np.ndarray): Data features
        costs (np.ndarray): Cost vectors
        P_r_windows (np.ndarray): Windows of renewable power generation data
        P_l_windows (np.ndarray): Windows of load demand data
        sols (np.ndarray): Optimal solutions
        objs (np.ndarray): Optimal objective values
    """

    def __init__(self, model, feats, costs, P_r_windows, P_l_windows):
        """
        Initialize an optDataset instance with dynamic updating for renewable generation and load profiles.

        Args:
            model (optModel): An instance of optModel
            feats (np.ndarray): Data features
            costs (np.ndarray): Costs of objective function
            P_r_windows (np.ndarray): Windows of renewable power generation data
            P_l_windows (np.ndarray): Windows of load demand data
        """
        if not isinstance(model, optModel):
            raise TypeError("arg model is not an optModel")
        self.model = model
        self.feats = feats
        self.costs = costs
        self.P_r_windows = P_r_windows
        self.P_l_windows = P_l_windows
        # find optimal solutions
        self.sols, self.objs = self._getSols()

    def _getSols(self):
        """
        A method to get optimal solutions for all cost vectors by dynamically updating the parameters.
        """
        sols = []
        objs = []
        print("Optimizing for optDataset...")
        time.sleep(1)
        for idx, cost in enumerate(tqdm(self.costs)):
            try:
                sol, obj = self._solve(cost, self.P_r_windows[idx], self.P_l_windows[idx])
            except:
                raise ValueError(
                    "For optModel, the method 'solve' should return solution vector and objective value."
                )
            sols.append(sol)
            objs.append([obj])
        return np.array(sols), np.array(objs)

    def _solve(self, cost, P_r, P_l):
        """
        Solve the optimization problem with given cost and dynamic parameters.

        Args:
            cost (np.ndarray): Cost of objective function
            P_r (np.ndarray): Current renewable generation data
            P_l (np.ndarray): Current load demand data

        Returns:
            tuple: Optimal solution (np.ndarray) and objective value (float)
        """
        self.model.update_parameters(P_r, P_l)
        self.model.setObj(cost)
        sol, obj = self.model.solve()
        return sol, obj

    def __len__(self):
        """
        Return the number of optimization problems.
        """
        return len(self.costs)

    def __getitem__(self, index):
        """
        Retrieve data by index.

        Args:
            index (int): Data index

        Returns:
            tuple: Data features (torch.tensor), costs (torch.tensor), optimal solutions (torch.tensor) and objective values (torch.tensor)
        """
        return (
            torch.FloatTensor(self.feats[index]),
            torch.FloatTensor(self.costs[index]),
            torch.FloatTensor(self.sols[index]),
            torch.FloatTensor(self.objs[index]),
        )
