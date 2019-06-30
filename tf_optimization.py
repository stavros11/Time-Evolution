"""Optimizes tensorflow models that inherit models/base.py.

Uses all states to perform the optimization.
Unless otherwise defined, it uses tensorflow automatic gradients for the
updates.

The model and the updater method should be defined in the script bellow.
If the model has gradients hard-coded, then the updater should be a function
that returns the complex gradients.
If we want to use auto-diff (the model has no hard-coded gradients) then
the updater should be a function that calculates the local energy loss.
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import utils
from energy import full_tf as en
from models import simple
tf.enable_v2_behavior()


# System parameters
n_sites = 4
time_steps = 100
t_final = 1.0
h_init = 1.0
h_ev = 0.5

# Optimization parameters
ctype = tf.complex128
n_epochs = 30000
n_message = 500
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)


# Find exact evolution state
t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)
exact_state_tf = tf.cast(exact_state, dtype=ctype)

# Define Hamiltonian matrices
ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)
ham = tf.cast(ham, dtype=ctype)
ham2 = tf.cast(ham2, dtype=ctype)


# Define TF model
#model = simple.RBMModel(exact_state[0], time_steps, n_hidden=8,
#                        rtype=tf.float64, ctype=tf.complex128)
model = simple.MPSModel(exact_state[0], time_steps, d_bond=4,
                        rtype=tf.float64, ctype=tf.complex128)
#model = simple.SequentialDenseModel(exact_state[0], time_steps)
#model = autoregressive.FullAutoregressiveModel(exact_state[0], time_steps)


# Define loss function for autograd or hardcoded complex gradients
# only one of the two cases can be used

# Case 1: updater is a function that returns Eloc and auto diff is used.
updater = lambda psi: tf.real(en.all_states_Eloc(psi, ham, dt, Ham2=ham2))

# Case 2: updater is a function that returns the complex gradients
#def updater(full_psi, dt=dt):
#  Ok, Ok_star_Eloc, Eloc = en.all_states_gradient(full_psi, ham, dt, Ham2=ham2)
#  return Ok_star_Eloc - tf.conj(Ok) * Eloc


# Optimize
overlaps = []
for epoch in range(n_epochs):
  model.update(optimizer, updater)
  overlaps.append(model.overlap(exact_state_tf, normalize_states=True).numpy())
  if epoch % n_message == 0:
    print("Epoch: {},  Overlap: {}".format(epoch, overlaps[-1]))

plt.plot(overlaps)