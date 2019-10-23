"""Base machine to be used for sampling calculations.

All machines should inherit this.
Machines are used when optimizing with sampling.
"""
import numpy as np
import tensorflow as tf
from typing import List, Optional


class BaseAutoGrad:
  """Models optimized by `deterministic_auto` should inherit this class.

  To create a new model, inherit this and implement the `forward` method.
  Optionally redefine `cast_time` if the model takes one-hot times as input.
  """

  def __init__(self, n_sites: int, time_steps: int,
               init_state: np.ndarray,
               input_type = tf.float32,
               optimizer: Optional[tf.train.Optimizer] = None):
    """Constructs machine given a keras model.

    Args:
      model_norm: Model that implements the wavefunction norm.
      model_norm: Model that implements the wavefunction phase.
        The complex wavefunction uses the polar complex form of norm and phase.
      n_sites: Number of spin sites.
      time_steps: Number of time steps (t=0 excluded).
      name: Name of the machine for saving.
      optimizer: TensorFlow optimizer to use for optimization.
    """
    # Time steps do not include initial condition
    self.n_sites = n_sites
    self.time_steps = time_steps
    self.dtype = None
    self.shape = None
    self.name = "keras"

    self._dense = None
    self.input_type = input_type
    self.dense_shape = (time_steps + 1, 2**n_sites)
    self.variables = []

    if optimizer is None:
      self.optimizer = tf.train.AdamOptimizer()
    else:
      self.optimizer = optimizer

    self.init_state = tf.convert_to_tensor(init_state[np.newaxis],
                                           dtype=self.output_type)

  @property
  def dense(self) -> np.ndarray:
    if self._dense is None:
      raise ValueError("Cannot call `machine.dense` before calling "
                       "`machine.wavefunction`.")
    return self._dense

  @property
  def output_type(self):
    if self.input_type == tf.float32:
      return tf.complex64
    elif self.input_type == tf.float64:
      return tf.complex128
    else:
      raise TypeError("Input type {} is unknown.".format(self.input_type))

  def add_variable(self, init_value: np.ndarray) -> tf.Tensor:
    var = tf.Variable(init_value, trainable=True, dtype=self.input_type)
    self.variables.append(var)
    return var

  def wavefunction(self) -> tf.Tensor:
    """Calculates wavefunction value on given samples.

    Note that currently this works only if configs and times are all the
    possible (exponentially many) states, to avoid using `where` in TensorFlow.
    We need to fix this if we want to use this with sampling.

    Args:
      configs: Spin configuration samples of shape (Ns, N)
      times: Time configuration samples of shape (Ns,)

    Returns:
      dense wavefunction of shape (T + 1, 2^N)
    """
    # TODO: Add a flag in `model` to select one-hot encoding for time
    # and apply this here
    psi = self.forward()
    self._dense = psi.numpy().reshape(self.dense_shape)
    return psi

  def forward(self) -> tf.Tensor:
    raise NotImplementedError

  def gradient(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Gradient method is not supported in AutoGrad "
                              "machines.")

  def update(self, grads: List[tf.Tensor]):
    self.optimizer.apply_gradients(zip(grads, self.variables))

  def update_time_step(self, new: np.ndarray, time_step: int):
    raise NotImplementedError("Step update is not supported in AutoGrad "
                              "machines.")