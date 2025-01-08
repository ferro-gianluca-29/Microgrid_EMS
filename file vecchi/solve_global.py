from pyomo.environ import SolverFactory
import matplotlib.pyplot as plt
import global_scheduling  # This assumes that all model setup is correct and complete within this module.
import data  # Import only if the global_scheduling module does not import it. If global_scheduling already imports it, this is not needed.

# Access model
model = global_scheduling.model

# Solve the model using a specific solver, e.g., GLPK
solver = SolverFactory('glpk')
solver.solve(model)

# Additional code to handle results, plotting, etc.
