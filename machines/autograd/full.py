import numpy as np
import tensorflow as tf
from machines.autograd import base


class FullWavefunctionModel(base.BaseAutoGrad):

  def __init__(self, **kwargs):
    init_state = kwargs["init_state"]
    super(FullWavefunctionModel, self).__init__(**kwargs)
    self.name = "fullwv_autograd"

    # Initialize full wavefunction by repeating the initial state
    init_value = np.array((self.time_steps + 1) * [init_state]).ravel()
    self.psi_re = self.add_variable(init_value.real)
    self.psi_im = self.add_variable(init_value.imag)

  def forward(self) -> tf.Tensor:
    psi = tf.complex(self.psi_re, self.psi_im)
    psi = tf.reshape(psi, self.dense_shape)
    return tf.concat([self.init_state, psi[1:]], axis=0)


class FullPropagatorModel(base.BaseAutoGrad):

  def __init__(self, **kwargs):
    super(FullPropagatorModel, self).__init__(**kwargs)
    self.name = "fullprop_autograd"
    self.n_states = 2**self.n_sites

    # Initialize propagator close to identity
    ident = np.eye(self.n_states)
    noise = np.random.normal(0.0, 1e-2, size=ident.shape)
    self.u_re = self.add_variable(ident + noise)
    noise = np.random.normal(0.0, 1e-2, size=ident.shape)
    self.u_im = self.add_variable(ident + noise)

  def forward(self) -> tf.Tensor:
    psis = [self.init_state[0][:, tf.newaxis]]
    u = tf.complex(self.u_re, self.u_im)
    for _ in range(self.time_steps):
      psis.append(tf.matmul(u, psis[-1]))
    psi = tf.stack(psis)[1:, :, 0]
    return tf.concat([self.init_state, psi], axis=0)