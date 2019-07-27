"""Quick tests of MPS machines.

To be deleted / moved, just put it on the main dir to use with spyder."""

import numpy as np
from machines import mps
import utils


# Model parameters
n_sites = 4
time_steps = 20
t_final = 1.0
h_init = 0.5
h_ev = 1.0

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)

machine = mps.SmallMPSMachine(exact_state[0], time_steps, d_bond=5)

dense_bin = np.copy(machine._dense)
dense_env = machine._create_envs()


n_samples = 1000
configs = np.random.randint(0, 2, size=(n_samples, n_sites))
times = np.random.randint(0, time_steps + 1, size=(n_samples,))

grad = machine.gradient(configs, times)