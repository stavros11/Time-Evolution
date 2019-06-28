"""Simple script to find ground states using the RBM ansatz.

To be used for initialization of the RBM for time evolution.
Assumes no biases.
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
import utils
tf.enable_v2_behavior()

n_sites = 4
n_hidden = 4
h = 1.0
n_epochs = 1000
n_message = 200
rtype = tf.float64
ctype = tf.complex128

def logrbm_forward(w, x):
  w_x = tf.matmul(x, w, transpose_b=True)
  return tf.reduce_sum(tf.cosh(w_x), axis=1)

def wavefunction(w_norm, w_phase, x):
  logpsi_re = logrbm_forward(w_norm, x)
  logpsi_im = tf.exp(logrbm_forward(w_phase, x))
  return tf.exp(tf.complex(logpsi_re, logpsi_im))


w_shape = (n_hidden, n_sites)
w_norm = tf.Variable(np.random.normal(0.0, 1e-3, size=w_shape),
                     dtype=rtype, trainable=True)
w_phase = tf.Variable(np.random.normal(0.0, 1e-3, size=w_shape),
                      dtype=rtype, trainable=True)
variables = [w_norm, w_phase]

all_states = tf.cast(np.array(list(itertools.product([-1, 1], repeat=n_sites))),
                     rtype)
ham = utils.tfim_hamiltonian(n_sites, h=h)
exact_energy = np.linalg.eigvalsh(ham)[0]
ham = tf.cast(utils.tfim_hamiltonian(n_sites, h=h), dtype=ctype)
optimizer = tf.train.AdamOptimizer()

energy_errors = []
for epoch in range(n_epochs):
  with tf.GradientTape() as tape:
    tape.watch(variables)
    psi = wavefunction(w_norm, w_phase, all_states)

    norm = tf.reduce_sum(tf.square(tf.abs(psi)))
    H_psi = tf.matmul(ham, psi[:, tf.newaxis])[:, 0]
    energy = tf.real(tf.reduce_sum(tf.conj(psi) * H_psi)) / norm

  grads = tape.gradient(energy, variables)
  optimizer.apply_gradients(zip(grads, variables))
  energy_errors.append(-np.abs(energy.numpy() - exact_energy) *
                        100.0 / exact_energy)

  if epoch % n_message == 0:
    print("Energy Error: {}%".format(energy_errors[-1]))

plt.semilogy(energy_errors)

np.save("rbm_init_weights/rbm_n{}h{}_w_norm.npy".format(n_sites, n_hidden),
        w_norm.numpy())
np.save("rbm_init_weights/rbm_n{}h{}_w_phase.npy".format(n_sites, n_hidden),
        w_phase.numpy())