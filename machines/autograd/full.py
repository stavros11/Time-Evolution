import numpy as np
import tensorflow as tf
from machines.autograd import base


class FullWavefunctionModel(base.BaseAutoGrad):

  def __init__(self, init_state: np.ndarray, time_steps: int,
               optimizer: tf.keras.optimizers.Optimizer):
    n_sites = int(np.log2(len(init_state)))
    name = "keras_fullwv"
    super(FullWavefunctionModel, self).__init__(n_sites, time_steps, name,
                                                optimizer)

    # Initialize full wavefunction by repeating the initial state
    init_value = np.array((self.time_steps + 1) * [init_state])
    self.psi_re = self.add_variable(init_value.real)
    self.psi_im = self.add_variable(init_value.imag)

    self.init_state = tf.convert_to_tensor(init_state[np.newaxis],
                                           dtype=self.ctype)
    self.bin2dec = tf.convert_to_tensor(2**np.arange(self.n_sites),
                                        dtype=tf.int32)[:, tf.newaxis]
    self._dense_tfcache = None

  def forward_dense(self) -> tf.Tensor:
    psi = tf.complex(self.psi_re, self.psi_im)
    psi = tf.concat([self.init_state, psi[1:]], axis=0)

    self._dense_cache = psi.numpy()
    self._dense_tfcache = tf.reshape(
        tf.math.log(psi), ((self.time_steps + 1) * self.n_states,))

    return psi

  def forward_log(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    if self._dense_tfcache is None:
      self.forward_dense()
    ids = tf.matmul(tf.math.maximum(0, x), self.bin2dec)[:, 0]
    ids += self.n_states * t
    return tf.gather(self._dense_tfcache, ids)

  def add_time_step(self):
    # FIXME: Update for new conventions
    self.variables = []
    self.time_steps += 1
    current_psi = self._dense

    new_shape = (self.time_steps + 1,) + current_psi.shape[1:]
    new_psi = np.zeros(new_shape, dtype=current_psi.dtype)
    new_psi[:-1] = np.copy(current_psi)
    new_psi[-1] = np.copy(current_psi[-1])

    self.psi_re = self.add_variable(new_psi.real)
    self.psi_im = self.add_variable(new_psi.imag)


class FullPropagatorModel(base.BaseAutoGrad):
  # FIXME: Update for new conventions

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

  def add_time_step(self):
    self.time_steps += 1