"""Base machine to be used for sampling calculations.

All machines should inherit this.
Machines are used when optimizing with sampling.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Optional


class BaseAutoGrad:

  def __init__(self, model_real: tf.keras.Model, model_imag: tf.keras.Model,
               n_sites: int, time_steps: int, use_mask: bool = False,
               name: Optional[str] = None,
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
    self.dtype = None # Type of the variational parameters
    self.shape = None # Shape of the variational parameters
    if name is None:
      self.name = "keras_composite"
    else:
      self.name = name

    self.real, self.imag = model_real, model_imag
    self.input_type = model_real.input.dtype
    input_shape = model_real.input.shape[-1]
    self.variables = (model_real.trainable_variables +
                      model_imag.trainable_variables)

    self._dense = None
    self.dense_shape = (time_steps + 1, 2**n_sites)

    if optimizer is None:
      self.optimizer = tf.train.AdamOptimizer()
    else:
      self.optimizer = optimizer

    if input_shape == self.n_sites + self.time_steps + 1:
      # one-hot encoding for time
      self.cast_time = self._one_hot_times
    elif input_shape == self.n_sites + 1:
      # single component encoding for time
      self.cast_time = lambda t: tf.cast(t, dtype=self.input_type)[:, tf.newaxis]
    else:
      raise ValueError("Model input dimensions are {} which is not supported "
                       "for given sites and time steps".format(input_shape))

    # Mask only used for `fullwv_model` for sanity check
    assert self.real.use_mask == self.imag.use_mask
    if self.real.use_mask:
      n_states = 2**n_sites
      grad_mask = np.ones(n_states * (time_steps + 1))
      grad_mask[:n_states] = 0
      self.grad_mask = tf.cast(grad_mask, dtype=self.input_type)
    else:
      self.grad_mask = None

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

  def wavefunction(self, configs: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
    """Calculates wavefunction value on given samples.

    For each time we have to calculate the wavefunction for the previous and
    next time step to calculate all energy terms!

    Args:
      configs: Spin configuration samples of shape (Ns, N)
      times: Time configuration samples of shape (Ns,)

    Returns:
      dense wavefunction of shape (Ns,)
    """
    configs_tf = tf.cast(configs, dtype=self.input_type)
    times_tf = self.cast_time(times)
    inputs = tf.concat([configs_tf, times_tf], axis=1)
    real, imag = self.real(inputs), self.imag(inputs)
    psi = tf.complex(real, imag)

    self._dense = psi.numpy().reshape(self.dense_shape)
    return psi

  def gradient(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Gradient method is not supported in AutoGrad "
                              "machines.")

  def update(self, grads: List[tf.Tensor]):
    if self.grad_mask is None:
      masked_grads = grads
    else:
      masked_grads = [g * self.grad_mask for g in grads]
    self.optimizer.apply_gradients(zip(masked_grads, self.variables))

  def update_time_step(self, new: np.ndarray, time_step: int):
    raise NotImplementedError("Step update is not supported in AutoGrad "
                              "machines.")


def feed_forward_network(n_input: int, n_output: int = 1) -> tf.keras.Model:
  """Simple feed forward keras model."""
  model = tf.keras.models.Sequential()
  model.add(layers.Input(n_input))
  model.add(layers.Dense(20, activation="relu"))
  model.add(layers.Dense(10, activation="relu"))
  model.add(layers.Dense(10, activation="relu"))
  model.add(layers.Dense(n_output, activation="sigmoid"))
  # Add an attribute that tells to the machine whether to use mask for to
  # keep the initial condition constant during optimization
  model.use_mask = False
  return model


def fullwv_model(init_wavefunction: np.ndarray, dtype=tf.float32) -> tf.keras.Model:
  n_states = init_wavefunction.shape[-1]
  n_sites = int(np.log2(n_states))
  to_binary = 2**np.arange(n_sites - 1, -1, -1)[:, np.newaxis]
  to_binary = tf.convert_to_tensor(to_binary, dtype=dtype)

  class FullWVLayer(layers.Layer):

    def __init__(self, init_value: np.ndarray, dtype=tf.float32):
      super(FullWVLayer, self).__init__()
      self.n_states = init_value.shape[-1]
      self.psi = tf.Variable(init_value.ravel(), trainable=True, dtype=dtype)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
      spins = (1 + inputs[:, :-1]) / 2
      spins = tf.cast(tf.matmul(spins, to_binary)[:, 0], dtype=tf.int32)
      times = tf.cast(inputs[:, -1], dtype=tf.int32)
      indices = spins + self.n_states * times
      return tf.gather(self.psi, indices)

  model = tf.keras.models.Sequential()
  model.add(layers.Input(n_sites + 1, dtype=dtype))
  model.add(FullWVLayer(init_wavefunction, dtype))
  # Add an attribute that tells to the machine whether to use mask for to
  # keep the initial condition constant during optimization
  model.use_mask = True
  return model