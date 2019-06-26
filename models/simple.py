"""Simple models."""

import numpy as np
import itertools
import tensorflow as tf
import utils
from models import base
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda


class FullStateModel(base.BaseModel):
  """Variational parameters are the full wavefunction as each time step."""

  def __init__(self, init_state, time_steps, std=1e-3,
               rtype=tf.float32, ctype=tf.complex64):
    state = np.array(time_steps * [init_state])
    state += utils.random_normal_complex(shape=state.shape, std=std)
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)
    self.real = tf.Variable(state.real, dtype=rtype, trainable=True)
    self.imag = tf.Variable(state.imag, dtype=rtype, trainable=True)

  def update(self, optimizer, updater):
    complex_grads = updater(self.wavefunction())
    grads = [tf.real(complex_grads), tf.imag(complex_grads)]
    variables = [self.real, self.imag]
    optimizer.apply_gradients(zip(grads, variables))

  def variational_wavefunction(self, training=False):
    return tf.complex(self.real, self.imag)


class SequentialDenseModel(base.BaseModel):
  """Simple sequential model with layers defined in _create_model method.

  Takes as input a spin configuration of shape (N,) and outputs the amplitudes
  for each time step with shape (M,) where M = time_steps. Two distinct NNs
  are used for the amplitude norms and the complex phases.
  TF autograd is used for training.
  """

  def __init__(self, init_state, time_steps,
               rtype=tf.float32, ctype=tf.complex64):
    n_sites = int(np.log2(len(init_state)))
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)

    self.all_confs = np.array(list(itertools.product([0, 1], repeat=n_sites)))
    self.all_confs = tf.cast(self.all_confs, dtype=rtype)

    self.model_norm = self._create_model(n_sites, time_steps)
    self.model_phase = self._create_model(n_sites, time_steps)
    self.vars = (self.model_norm.trainable_variables +
                 self.model_phase.trainable_variables)

  @staticmethod
  def _create_model(d_in, d_out):
    return tf.keras.Sequential([Input(shape=(d_in,)),
                                Dense(d_out // 2),
                                Dense(d_out)])

  def variational_wavefunction(self, training=False):
    norm = self.model_norm(self.all_confs, training=training)
    phase = self.model_phase(self.all_confs, training=training)
    psi = tf.complex(norm * tf.cos(2 * np.pi * phase),
                     norm * tf.sin(2 * np.pi * phase))
    return tf.transpose(psi, [1, 0])


class SequentialLSTMModel(SequentialDenseModel):

  @staticmethod
  def _create_model(d_in, d_out):
    add_dim = Lambda(lambda x: tf.expand_dims(x, axis=-1))
    return tf.keras.Sequential([Input(shape=(d_in,)), add_dim,
                                LSTM(10, return_sequences=True),
                                LSTM(d_out)])