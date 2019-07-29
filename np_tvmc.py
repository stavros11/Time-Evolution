"""Evolves wavefunction using traditional t-VMC.

Uses all states to calculate gradients.
"""

#import h5py
import numpy as np
import matplotlib.pyplot as plt
import utils
from energy import tvmc
from machines import mps


n_sites = 4
time_steps = 20
t_final = 1.0
h_init = 1.0
h_ev = 0.5
n_epochs = 10000
n_message = 500

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
#ham2 = ham.dot(ham)
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)

# Initialize machine
machine = mps.SmallMPSStepMachine(exact_state[0], d_bond=5)

history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
full_psi = [machine.dense()]
for step in range(time_steps):
  rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok = tvmc.exact_tvmc_step(machine,
                                                                 h=h_ev)

  machine.update(-1j * rhs * dt)
  full_psi.append(machine.dense())

full_psi = np.array(full_psi)
print(utils.averaged_overlap(full_psi, exact_state))