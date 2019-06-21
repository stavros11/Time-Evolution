"""Script for testing various modules."""

import numpy as np
import matplotlib.pyplot as plt
import calculate_energy_full as en
import utils


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

# Initialize variational state by copying initial condition and adding some noise
var_state = np.array((time_steps + 1) * [exact_state[0]])
var_state[1:] += utils.random_normal_complex(var_state[1:].shape, std=1e-2)

optimizer = utils.AdamComplex(var_state[1:].shape, dtype=var_state.dtype)

overlaps = []
for epoch in range(n_epochs):
  Ok, Ok_star_Eloc, Eloc, _ = en.all_states_gradient(np, var_state, ham,
                                                     dt, Ham2=ham2)
  complex_grad = Ok_star_Eloc - Ok.conj() * Eloc
  var_state[1:] += optimizer.update(complex_grad, epoch)
  
  overlaps.append(utils.overlap(var_state, exact_state))
  if epoch % 1000 == 0:
    print("Overlap: {}".format(overlaps[-1]))


plt.plot(np.arange(n_epochs), overlaps)
plt.show()  