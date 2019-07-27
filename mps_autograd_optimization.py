"""Autograd MPS optimization using contractions."""

import numpy as np
import tensorflow as tf
import utils
from energy import autograd_mps
from machines import mps_utils
tf.enable_v2_behavior()


# Model parameters
n_sites = 6
time_steps = 20
t_final = 1.0
h_init = 1.0
h_ev = 0.5

# MPS parameters
d_bond = 6
dtype = tf.float64

# Optimization parameters
n_epochs = 10000
n_message = 1
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]


# Calculate exact evolution
ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)


# Create TFIM MPOs
ham_mpo = tf.cast(utils.tfim_mpo(n_sites, h=h_ev), dtype=tf.complex128)
#ham2_mpo_shape = list(ham_mpo.shape)
#ham2_mpo_shape[1] = ham2_mpo_shape[1]**2
#ham2_mpo_shape[-1] = ham2_mpo_shape[-1]**2
#ham2_mpo = np.einsum("iLumR,ilmdr->iLludRr", ham_mpo, ham_mpo)
#ham2_mpo = ham2_mpo.reshape(ham2_mpo_shape)


# Write initial state as MPS for initialization.
mps_init = mps_utils.dense_to_mps(exact_state[0], d_bond=d_bond).swapaxes(1, 2)
mps_vars = [tf.Variable((time_steps + 1) * [mps_init.real], dtype=dtype),
            tf.Variable((time_steps + 1) * [mps_init.imag], dtype=dtype)]

# Create gradient mask so that we do not update the initial condition
grad_mask = np.stack([np.zeros_like(mps_init)] +
                      time_steps * [np.ones_like(mps_init)])
grad_mask = tf.cast(grad_mask, dtype=dtype)


# Optimize
history = {"exact_Eloc": [], "norms": []}
for epoch in range(n_epochs):
  with tf.GradientTape() as tape:
    tape.watch(mps_vars)
    mps = tf.complex(mps_vars[0], mps_vars[1])
    clock, energy, norms = autograd_mps.clock_energy(mps, ham_mpo, dt)

  grads = tape.gradient(tf.real(clock), mps_vars)
  grads = [grad_mask * g for g in grads]

  history["exact_Eloc"].append(clock.numpy())
  history["norms"].append(norms.numpy())
  if epoch % n_message == 0:
    print("\nEpoch: {}".format(epoch))
    print("Loss: {}".format(history["exact_Eloc"]))
    #print("Overlap: {}".format(history["overlaps"][-1]))
    #print("Exact Eloc: {}".format(history["exact_Eloc"][-1]))

