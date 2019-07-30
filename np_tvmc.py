"""Evolves wavefunction using traditional t-VMC.

Uses all states to calculate gradients.
"""

import os
import ctypes
import h5py
import numpy as np
import utils
from energy import tvmc
from machines import mps
from numpy.ctypeslib import ndpointer


# Model parameters
n_sites = 6
time_steps = 200
t_final = 3.0
h_init = 1.0
h_ev = 0.5
n_epochs = 10000
n_message = 500
bond_list = [2, 3, 4, 6]

# Sampling parameters (per time when using the space only sampler)
sampler = ctypes.CDLL(os.path.join(os.getcwd(), "samplers", "spacevmclib.so"))
sampler.run.argtypes = ([ndpointer(np.complex128, flags="C_CONTIGUOUS")] +
                         6 * [ctypes.c_int] +
                         [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")])
sampler.run.restype = None
sampler.n_samples = 5000
sampler.n_corr = 1
sampler.n_burn = 10

sampler = None

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
#ham2 = ham.dot(ham)
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)

history = {"overlaps" : [], "avg_overlaps": []}
for d_bond in bond_list:
  machine = mps.SmallMPSStepMachine(exact_state[0], d_bond=d_bond)
  full_psi = tvmc.evolve(machine, time_steps, dt, h=h_ev, sampler=sampler)

  history["overlaps"].append(utils.overlap(full_psi, exact_state))
  history["avg_overlaps"].append(utils.averaged_overlap(full_psi, exact_state))

  print("\nD: {}".format(d_bond))
  print("Overlap: {}".format(history["overlaps"][-1]))
  print("Averaged Overlap: {}".format(history["avg_overlaps"][-1]))

  # Save dense
  if sampler is None:
    filename = "tvmc_allstates_{}_N{}M{}.npy".format(
        machine.name, n_sites, time_steps)
  else:
    filename = "tvmc_sampling{}_{}_N{}M{}.npy".format(
        sampler.n_samples, machine.name, n_sites, time_steps)
  np.save("final_dense/{}".format(filename), full_psi)


if sampler is None:
  filename = "tvmc_allstates_mpsd{}_{}_N{}M{}".format(
      bond_list[0], bond_list[-1], n_sites, time_steps)
else:
  filename = "tvmc_sampling{}_mpsd{}_{}_N{}M{}".format(
      sampler.n_samples, bond_list[0], bond_list[-1], n_sites, time_steps)
file = h5py.File("histories/{}.h5py".format(filename), "w")
for k in history.keys():
  file[k] = history[k]
file.close()
