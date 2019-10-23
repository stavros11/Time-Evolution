import itertools
import numpy as np
import tensorflow as tf
from machines.autograd import base
from tensorflow.keras import layers
from typing import List, Tuple


class StepConvModel(base.BaseAutoGrad):
  # Very slow!

  def __init__(self, **kwargs):
    super(StepConvModel, self).__init__(**kwargs)
    self.name = "step_conv"

    self.models_re = [self.create_model() for _ in range(self.time_steps)]
    self.models_im = [self.create_model() for _ in range(self.time_steps)]

    all_confs = np.array(list(itertools.product([0, 1], repeat=self.n_sites)))
    self.all_confs = tf.convert_to_tensor(all_confs, dtype=self.input_type)
    self.all_confs = self.all_confs[:, :, tf.newaxis]

  def create_model(self) -> tf.keras.models.Model:
    model = tf.keras.models.Sequential()
    model.add(layers.Input((self.n_sites, 1)))
    model.add(layers.Conv1D(filters=8, kernel_size=3,
                            padding="same", activation="relu"))
    model.add(layers.MaxPooling1D())
    model.add(layers.Conv1D(filters=4, kernel_size=3,
                            padding="same", activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="tanh"))
    self.variables += model.variables
    return model

  def forward(self) -> tf.Tensor:
    psi = [self.init_state[0]]
    for model_re, model_im in zip(self.models_re, self.models_im):
      re = model_re(self.all_confs)
      im = model_im(self.all_confs)
      psi.append(tf.complex(re, im)[:, 0])
    return tf.stack(psi)


class StepFeedForwardModel(base.BaseAutoGrad):

  def __init__(self, **kwargs):
    super(StepFeedForwardModel, self).__init__(**kwargs)
    self.name = "step_ffnn"

    all_confs = list(itertools.product([0, 1], repeat=self.n_sites))
    all_confs = np.array(self.time_steps * [all_confs]).swapaxes(0, 1)
    self.all_confs = tf.convert_to_tensor(all_confs, dtype=self.input_type)

    self._next_input_units = self.n_sites
    self.real_params = [self.add_layer_weights(16),
                        self.add_layer_weights(8),
                        self.add_layer_weights(1)]
    self._next_input_units = self.n_sites
    self.imag_params = [self.add_layer_weights(16),
                        self.add_layer_weights(8),
                        self.add_layer_weights(1)]

  def add_layer_weights(self, units: int) -> Tuple[tf.Tensor, tf.Tensor]:
    shape = (1, self.time_steps, units, self._next_input_units)
    w = self.add_variable(np.random.normal(0.0, 1e-3, size=shape))
    b = self.add_variable(np.zeros(shape[:-1]))
    self._next_input_units = units
    return w, b

  @staticmethod
  def layer_forward(params: Tuple[tf.Tensor, tf.Tensor],
                    inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of a (hard-coded) feed-forward network.

    Args:
      inputs: Input to the layer with shape (n_states, time_steps, fan_in).
      params: Tuple with weights and biases for the layer.
        weights have shape (1, time_steps, fan_out, fan_in).
        biases have shape (1, time_steps, fan_out)

    Returns:
      Result of the affine transformation with shape
        (n_states, time_steps, fan_out).
    """
    w, b = params
    prod = tf.matmul(w, inputs[:, :, :, tf.newaxis])[:, :, :, 0]
    return prod + b

  def model_forward(self, param_list: List[Tuple[tf.Tensor, tf.Tensor]],
                    inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of a whole model."""
    act = self.layer_forward(param_list[0], inputs)
    for params in param_list[1:]:
      act = tf.nn.relu(act)
      act = self.layer_forward(params, act)
    return tf.nn.tanh(act)

  def forward(self) -> tf.Tensor:
    psi_re = self.model_forward(self.real_params, self.all_confs)
    psi_im = self.model_forward(self.imag_params, self.all_confs)
    psi = tf.transpose(tf.complex(psi_re, psi_im)[:, :, 0])
    return tf.concat([self.init_state, psi], axis=0)