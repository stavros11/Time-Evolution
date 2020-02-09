import numpy as np
import tensorflow as tf


class FullWavefunction:

  def __init__(self, initial_state: np.ndarray, dtype=tf.float64,
               learning_rate: float = 1e-3):
    self.rtype = dtype
    self.var_re = tf.Variable(initial_state.real, dtype=self.rtype)
    self.var_im = tf.Variable(initial_state.imag, dtype=self.rtype)
    self.variables = [self.var_re, self.var_im]

    self._dense = None
    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    mask = np.ones_like(initial_state)
    mask[0] = 0.0
    self.mask = tf.convert_to_tensor(mask, dtype=dtype)

  def __call__(self) -> tf.Tensor:
    return tf.complex(self.var_re, self.var_im)

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
      self._dense = tf.complex(self.model_re(), self.model_im())
    return self._dense

  def update(self, objective) -> tf.Tensor:
    with tf.GradientTape() as tape:
      self._dense = self.__call__()
      loss = objective(self._dense)

    grads = tape.gradient(loss, self.variables)
    grads = [self.mask * g for g in grads]
    self.optimizer.apply_gradients(zip(grads, self.variables))
    return loss
