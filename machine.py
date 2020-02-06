import numpy as np
import tensorflow as tf


class CartesianMachine:

  def __init__(self,
               model_re: tf.keras.Model,
               model_im: tf.keras.Model,
               learning_rate: float = 1e-3):
    if model_re.dtype != model_im.dtype:
      raise TypeError("Real and imaginary models have different types.")
    self.rtype = model_re.dtype

    self.model_re = model_re
    self.model_im = model_im

    self.optimizer_re = tf.keras.optimizers.Adam(learning_rate)
    self.optimizer_im = tf.keras.optimizers.Adam(learning_rate)
    self._dense = None

  @property
  def dense(self):
    if self._dense is None:
      self._dense = tf.complex(self.model_re(), self.model_im())
    return self._dense

  @property
  def ctype(self):
    if self.rtype == tf.float32:
      return tf.complex64
    elif self.rtype == tf.float64:
      return tf.complex128
    else:
      raise TypeError

  def update(self, objective) -> tf.Tensor:
    with tf.GradientTape() as tape:
      self._dense = tf.complex(self.model_re(), self.model_im())
      loss = objective(self._dense)

    n = len(self.model_re.variables)
    variables = self.model_re.variables + self.model_im.variables
    grads = tape.gradient(loss, variables)
    grads_re = self.model_re.mask_grads(grads[:n])
    grads_im = self.model_im.mask_grads(grads[n:])

    self.optimizer_re.apply_gradients(zip(grads_re, self.model_re.variables))
    self.optimizer_im.apply_gradients(zip(grads_im, self.model_im.variables))
    return loss


class FullWavefunction:

  def __init__(self, initial_state: np.ndarray, dtype=tf.float64):
    self.dtype = dtype
    self.psi = tf.Variable(initial_state, dtype=dtype)
    self.variables = [self.psi]

    mask = np.ones_like(initial_state)
    mask[0] = 0.0
    self.mask = tf.convert_to_tensor(mask, dtype=dtype)

  def __call__(self) -> tf.Tensor:
    return self.psi

  def mask_grads(self, grad: tf.Tensor) -> tf.Tensor:
    return self.mask * grad
