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
time_steps = 20
t_final = 1.0
h_init = 1.0
h_ev = 0.5
n_epochs = 10000
n_message = 500
init_prod = False
fullwv = True
norm_clock = True

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)

if init_prod:
  init_state = np.ones(2**n_sites) / np.sqrt(2**n_sites)
  exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                                h=h_ev,
                                                init_state=init_state)
else:
  exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                                h0=h_init, h=h_ev)

# Initialize machine
if fullwv:
  machine = full.FullWavefunctionMachine(exact_state[0], time_steps)
else:
  machine = mps.SmallMPSMachine(exact_state[0], time_steps, d_bond=3)
optimizer = utils.AdamComplex(machine.shape, dtype=machine.dtype)

history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
full_psi = machine.dense()
for epoch in range(n_epochs):
  if fullwv:
    Ok, Ok_star_Eloc, Eloc, _ = full_np.all_states_gradient(full_psi, ham, dt,
                                                            norm=norm_clock,
                                                            Ham2=ham2)
  else:
    Ok, Ok_star_Eloc, Eloc, _ = full_np.all_states_sampling_gradient(
        machine, ham, dt, norm=norm_clock, Ham2=ham2)

  grad = Ok_star_Eloc - Ok.conj() * Eloc
  if grad.shape[1:] != machine.shape[1:]:
    grad = grad.reshape((time_steps,) + machine.shape[1:])

  machine.update(optimizer.update(grad, epoch))
  full_psi = machine.dense()

  #if epoch % 50 == 0:
  #  print((np.abs(full_psi)**2).sum(axis=1))

  history["exact_Eloc"].append(Eloc)
  history["overlaps"].append(utils.overlap(full_psi, exact_state))
  history["avg_overlaps"].append(utils.averaged_overlap(full_psi, exact_state))
  if epoch % n_message == 0:
    print("\nEpoch {}".format(epoch))
    print("Eloc: {}".format(history["exact_Eloc"][-1]))
    print("Overlap: {}".format(history["overlaps"][-1]))
    print("Averaged Overlap: {}".format(history["avg_overlaps"][-1]))

# Save history
if init_prod:
  filename = "initprod_allstates_{}_N{}M{}.h5py".format(machine.name, n_sites, time_steps)
else:
  filename = "allstates_{}_N{}M{}.h5py".format(machine.name, n_sites, time_steps)
file = h5py.File("histories/{}".format(filename), "w")
for k in history.keys():
  file[k] = history[k]
file.close()

# Save final dense wavefunction
filename = "{}.npy".format(filename[:-5])
np.save("final_dense/{}".format(filename), full_psi)
