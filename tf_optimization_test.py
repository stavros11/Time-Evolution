"""Script for testing various modules."""

import numpy as np
import matplotlib.pyplot as plt
import calculate_energy_tf as en
import tensorflow as tf
import utils
tf.enable_v2_behavior()


class VariationalState():
  def __init__(self, state, rtype=tf.float64, ctype=tf.complex128):
    self.init_state = tf.cast(state[0][np.newaxis], dtype=ctype)
    self.real = tf.Variable(state[1:].real, dtype=rtype, trainable=True)
    self.imag = tf.Variable(state[1:].imag, dtype=rtype, trainable=True)
  
  def update(self, optimizer, complex_grads):
    grads = [tf.real(complex_grads), tf.imag(complex_grads)]
    variables = [self.real, self.imag]
    optimizer.apply_gradients(zip(grads, variables))
    
  def full(self):
    state_t = tf.complex(self.real, self.imag)
    return tf.concat([self.init_state, state_t], axis=0)
  
  def numpy(self):
    return self.full().numpy()
    

n_sites = 4
time_steps = 100
t_final = 1.0
h_init = 0.5
h_ev = 1.0
n_epochs = 12000

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)
ham = tf.cast(ham, dtype=tf.complex128)
ham2 = tf.cast(ham2, dtype=tf.complex128)

var_state = np.array((time_steps + 1) * [exact_state[0]])
var_state[1:] += utils.random_normal_complex(var_state[1:].shape, std=1e-2)

var_state = VariationalState(var_state)
optimizer = tf.train.AdamOptimizer()


overlaps = []
for epoch in range(n_epochs):
  Ok, Ok_star_Eloc, Eloc = en.all_states_gradient(var_state.full(), ham, dt, 
                                                  Ham2=ham2)
  complex_grad = Ok_star_Eloc - tf.conj(Ok) * Eloc
  # var_state[1:] += optimizer.update(complex_grad, epoch)
  var_state.update(optimizer, complex_grad)
  
  overlaps.append(utils.overlap(var_state.numpy(), exact_state))
  if epoch % 1000 == 0:
    print("Overlap: {}".format(overlaps[-1]))


plt.plot(np.arange(n_epochs), overlaps)
plt.show()