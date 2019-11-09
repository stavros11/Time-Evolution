"""Base machine to be used for sampling calculations.

All machines should inherit this.
Machines are used when optimizing with sampling.
"""
import numpy as np
import tensorflow as tf
from typing import List


class BaseAutoGrad:
  """Models optimized by `deterministic_auto` should inherit this class.

  To create a new model, inherit this and implement the `forward` method.
  Optionally redefine `cast_time` if the model takes one-hot times as input.
  """

  def __init__(self, n_sites: int, time_steps: int, name: str,
               optimizer: tf.keras.optimizers.Optimizer):
    # Time steps do not include initial condition
    self.rtype = None
    self.n_sites = n_sites
    self.n_states = 2 ** n_sites
    self.time_steps = time_steps
    self.name = name

    self.variables = []
    self.optimizer = optimizer

    self._forward_log = None # np.ndarray
    self._dense_cache = None # np.ndarray

  def forward_log(self, x: tf.Tensor, t: tf.Tensor) -> tf.Tensor:
    """HAS TO BE IMPLEMENTED WHEN INHERITING"""
    self._forward_log = None # np.ndarray
    raise NotImplementedError

  def forward_dense(self) -> tf.Tensor:
    """HAS TO BE IMPLEMENTED WHEN INHERITING"""
    self._dense_cache = None # np.ndarray
    raise NotImplementedError

  def update(self, grads: List[tf.Tensor]):
    self.optimizer.apply_gradients(zip(grads, self.variables))
    self._forward_cache = None
    self._dense_cache = None

  def add_variable(self, init_value: np.ndarray) -> tf.Tensor:
    tftype = self.np_to_tf_type(init_value.dtype)
    if self.rtype is None:
      self.rtype = tftype
    else:
      assert tftype == self.rtype

    var = tf.Variable(init_value, trainable=True, dtype=self.rtype)
    self.variables.append(var)
    return var

  @property
  def ctype(self):
    if self.rtype == tf.float32:
      return tf.complex64
    elif self.rtype == tf.float64:
      return tf.complex128
    else:
      raise TypeError("Unknown complex type.")

  @staticmethod
  def np_to_tf_type(nptype):
    if nptype == np.float32:
      return tf.float32
    elif nptype == np.float64:
      return tf.float64
    else:
      raise TypeError("Cannot conver numpy type to tensorflow.")

  @property
  def dense(self) -> np.ndarray:
    if self._dense_cache is None:
      self.forward_dense()
    return self._dense_cache

  @property
  def log(self) -> np.ndarray:
    if self._forward_log is None:
      raise ValueError("Log cache is unavailable before calling forward.")
    return self._forward_log