"""Base machine to be used for sampling calculations.

All machines should inherit this.
Machines are used when optimizing with sampling.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Optional


class BaseAutoGrad:
  """Models optimized by `deterministic_auto` should inherit this class.

  To create a new model, inherit this and implement the `real` and `imag`
  methods. Optionally redefine `cast_time` if the model takes one-hot times
  as input.
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

  def wavefunction(self, configs: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
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
    configs_tf = tf.cast(configs, dtype=self.input_type)
    times_tf = tf.cast(times, dtype=self.input_type)
    psi_forward = self.forward(configs_tf, times_tf)
    psi_ev = tf.reshape(psi_forward, self.dense_shape)
    psi = tf.concat([self.init_state, psi_ev[1:]], axis=0)

    self._dense = psi.numpy().reshape(self.dense_shape)
    return psi

  def forward(self, configs: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError

  def gradient(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Gradient method is not supported in AutoGrad "
                              "machines.")

  def update(self, grads: List[tf.Tensor]):
    self.optimizer.apply_gradients(zip(grads, self.variables))

  def update_time_step(self, new: np.ndarray, time_step: int):
    raise NotImplementedError("Step update is not supported in AutoGrad "
                              "machines.")


class FullWavefunctionModel(BaseAutoGrad):

  def __init__(self, **kwargs):
    init_state = kwargs["init_state"]
    super(FullWavefunctionModel, self).__init__(**kwargs)
    self.name = "fullwv_autograd"
    self.n_states = 2**self.n_sites
    self.to_binary = 2**np.arange(self.n_sites - 1, -1, -1)[:, np.newaxis]
    self.to_binary = tf.convert_to_tensor(self.to_binary, dtype=self.input_type)

    # Initialize full wavefunction by repeating the initial state
    init_value = np.array((self.time_steps + 1) * [init_state]).ravel()
    self.psi_re = self.add_variable(init_value.real)
    self.psi_im = self.add_variable(init_value.imag)

  def call(self, spins: tf.Tensor, times: tf.Tensor, psi: tf.Tensor) -> tf.Tensor:
    spins = (1 + spins) / 2
    spins = tf.cast(tf.matmul(spins, self.to_binary)[:, 0], dtype=tf.int32)
    times = tf.cast(times, dtype=tf.int32)
    indices = spins + self.n_states * times
    return tf.gather(psi, indices)

  def real(self, configs: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
    return self.call(configs, times, self.psi_re)

  def imag(self, configs: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
    return self.call(configs, times, self.psi_im)

  def forward(self, configs: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
    re = self.real(configs, times)
    im = self.imag(configs, times)
    return tf.complex(re, im)


class FullPropagatorModel(BaseAutoGrad):

  def __init__(self, **kwargs):
    super(FullPropagatorModel, self).__init__(**kwargs)
    self.name = "fullprop_autograd"
    self.n_states = 2**self.n_sites
    self.n_clock_states = self.n_states * (self.time_steps + 1)
    self.to_binary = 2**np.arange(self.n_sites - 1, -1, -1)[:, np.newaxis]
    self.to_binary = tf.convert_to_tensor(self.to_binary, dtype=self.input_type)

    ident = np.eye(self.n_states)
    noise = np.random.normal(0.0, 1e-2, size=ident.shape)
    self.u_re = self.add_variable(ident + noise)
    noise = np.random.normal(0.0, 1e-2, size=ident.shape)
    self.u_im = self.add_variable(ident + noise)

  def calculate_psis(self) -> tf.Tensor:
    psis = [self.init_state[0][:, tf.newaxis]]
    u = tf.complex(self.u_re, self.u_im)
    for _ in range(self.time_steps):
      psis.append(tf.matmul(u, psis[-1]))
    return tf.reshape(tf.stack(psis), (self.n_clock_states,))

  def call(self, spins: tf.Tensor, times: tf.Tensor, psi: tf.Tensor) -> tf.Tensor:
    spins = (1 + spins) / 2
    spins = tf.cast(tf.matmul(spins, self.to_binary)[:, 0], dtype=tf.int32)
    times = tf.cast(times, dtype=tf.int32)
    indices = spins + self.n_states * times
    return tf.gather(psi, indices)

  def forward(self, configs: tf.Tensor, times: tf.Tensor) -> tf.Tensor:
    psis = self.calculate_psis()
    return self.call(configs, times, psis)


def feed_forward_model(n_input: int, n_output: int = 1, dtype=tf.float32) -> tf.keras.Model:
  """Simple feed forward keras model."""
  # THIS IS DEPRECATED!
  model = tf.keras.models.Sequential()
  model.add(layers.Input(n_input, dtype=dtype))
  model.add(layers.Dense(20, activation="relu", dtype=dtype))
  model.add(layers.Dense(20, activation="relu", dtype=dtype))
  model.add(layers.Dense(n_output, activation="sigmoid", dtype=dtype))
  return model