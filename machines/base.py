import numpy as np
import tensorflow as tf


class Base:

  def __init__(self, dtype=tf.float64, learning_rate: float = 1e-3):
    self.rtype = dtype
    self._dense = None
    self.mask = None
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

  @property
  def ctype(self):
    if self.rtype == tf.float32:
      return tf.complex64
    elif self.rtype == tf.float64:
      return tf.complex128
    else:
      raise TypeError

  @property
  def dense(self):
    if self._dense is None:
      self._dense = self.__call__()
    return self._dense

  def update(self, objective) -> tf.Tensor:
    with tf.GradientTape() as tape:
      self._dense = self.__call__()
      loss = objective(self._dense)

    grads = tape.gradient(loss, self.variables)
    if self.mask is not None:
      grads = [self.mask * g for g in grads]
    self.optimizer.apply_gradients(zip(grads, self.variables))
    return loss


class FullWavefunction(Base):

  def __init__(self, initial_state: np.ndarray, dtype=tf.float64,
               learning_rate: float = 1e-3):
    super(FullWavefunction, self).__init__(dtype, learning_rate)
    self.var_re = tf.Variable(initial_state.real, dtype=self.rtype)
    self.var_im = tf.Variable(initial_state.imag, dtype=self.rtype)
    self.variables = [self.var_re, self.var_im]

    mask = np.ones_like(initial_state)
    mask[0] = 0.0
    self.mask = tf.convert_to_tensor(mask, dtype=dtype)

  def __call__(self) -> tf.Tensor:
    return tf.complex(self.var_re, self.var_im)


class FullProp(Base):

  def __init__(self, initial_state: np.ndarray, dtype=tf.float64,
               learning_rate: float = 1e-3):
    super(FullProp, self).__init__(dtype, learning_rate)
    self.time_steps, n_states = initial_state.shape
    self.time_steps -= 1

    self.u_re = tf.Variable(np.eye(n_states), dtype=self.rtype)
    self.u_im = tf.Variable(np.eye(n_states), dtype=self.rtype)
    self.variables = [self.u_re, self.u_im]

    self.psi0 = tf.convert_to_tensor(initial_state[0])

  def __call__(self):
    states = [self.psi0[:, tf.newaxis]]
    u = tf.complex(self.u_re, self.u_im)
    for _ in range(self.time_steps):
      states.append(tf.matmul(u, states[-1]))
    return tf.stack(states)[:, :, 0]
