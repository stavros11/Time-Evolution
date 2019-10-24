import numpy as np
from machines.autograd import rbm
from utils import tfim


n_sites = 6
time_steps = 20
t_final = 1.0

t_grid = np.linspace(0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]
exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                           h0=1.0, h=0.5)


w, b, c = rbm.fit_rbm_to_dense(exact_state[0], n_hidden=n_sites)