"""Optimizes full wavefunction using numpy.

Uses all states to calculate gradients.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import utils
from energy import full_np
from machines import full, mps


n_sites = 6
time_steps = 35
t_final = 1.0
h_init = 1.0
h_ev = 0.5
n_epochs = 10000
n_message = 500

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)

# Initialize machine
machine = full.FullWavefunctionMachine(exact_state[0], time_steps)
#machine = mps.SmallMPSMachine(exact_state[0], time_steps, d_bond=6)
optimizer = utils.AdamComplex(machine.shape, dtype=machine.dtype)

history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
full_psi = machine.dense()
for epoch in range(n_epochs):
  Ok, Ok_star_Eloc, Eloc, _ = full_np.all_states_gradient(full_psi,
                                                                   ham,
                                                                   dt,
                                                                   Ham2=ham2)

  grad = Ok_star_Eloc - Ok.conj() * Eloc
  if grad.shape[1:] != machine.shape[1:]:
    grad = grad.reshape((time_steps,) + machine.shape[1:])

  machine.update(optimizer.update(grad, epoch))
  #machine.update(-grad  / np.sqrt((np.abs(grad)**2).mean()))
  full_psi = machine.dense()

  history["exact_Eloc"].append(Eloc)
  history["overlaps"].append(utils.overlap(full_psi, exact_state))
  history["avg_overlaps"].append(utils.averaged_overlap(full_psi, exact_state))
  if epoch % n_message == 0:
    print("\nEpoch {}".format(epoch))
    print("Eloc: {}".format(history["exact_Eloc"][-1]))
    print("Overlap: {}".format(history["overlaps"][-1]))
    print("Averaged Overlap: {}".format(history["avg_overlaps"][-1]))

# Save history
filename = "allstates_{}_N{}M{}.h5py".format(machine.name, n_sites, time_steps)
file = h5py.File("histories/{}".format(filename), "w")
for k in history.keys():
  file[k] = history[k]
file.close()

# Save final dense wavefunction
filename = "{}.npy".format(filename[:-4])
np.save("final_dense/{}".format(filename), full_psi)