"""Optimizes full wavefunction using numpy.

Uses all states to calculate gradients.
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
from energy import full_np
from machines import full


n_sites = 4
time_steps = 100
t_final = 1.0
h_init = 0.5
h_ev = 1.0
n_epochs = 12000

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)

# Initialize machine
machine = full.FullWavefunctionMachine(exact_state[0], time_steps)
optimizer = utils.AdamComplex(machine.shape, dtype=machine.dtype)

overlaps = []
full_psi = machine.dense()
for epoch in range(n_epochs):
  Ok, Ok_star_Eloc, Eloc, _ = full_np.all_states_sampling_gradient(machine,
                                                                   ham,
                                                                   dt,
                                                                   Ham2=ham2)

  grad = Ok_star_Eloc - Ok.conj() * Eloc
  if grad.shape[1:] != machine.shape[1:]:
    grad = grad.reshape((time_steps,) + machine.shape[1:])

  machine.update(optimizer.update(grad, epoch))
  full_psi = machine.dense()

  overlaps.append(utils.overlap(full_psi, exact_state))
  if epoch % 1000 == 0:
    print("Overlap: {}".format(overlaps[-1]))


plt.plot(np.arange(n_epochs), overlaps)
plt.show()