import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from machines import base


class CopiedFFNN(base.Base):

  def __init__(self, initial_state: np.ndarray, n_hidden: int,
               dtype=tf.float32, learning_rate: float = 1e-3):
    super(CopiedFFNN, self).__init__(dtype, learning_rate)
    self.time_steps, n_states = initial_state.shape
    self.n_sites = int(np.log2(n_states))
    self.time_steps -= 1
    self.n_hidden = n_hidden

    self.model_re = self._create_model()
    self.model_im = self._create_model()
    self.variables = self.model_re.variables + self.model_im.variables

    all_confs = np.array(list(itertools.product([0, 1], repeat=self.n_sites)))
    self.all_confs = tf.convert_to_tensor(all_confs, dtype=self.rtype)

    self.psi0 = tf.convert_to_tensor(initial_state[0], dtype=self.ctype)
    self.psi0 = self.psi0[tf.newaxis] # (1, n_states)

  def _create_model(self) -> tf.keras.Model:
    model = tf.keras.models.Sequential()
    model.add(layers.Input((self.n_sites,)))
    n_hidden_total = self.time_steps * self.n_hidden
    model.add(layers.Dense(n_hidden_total, activation="relu"))
    model.add(layers.Reshape((n_hidden_total, 1)))
    model.add(layers.LocallyConnected1D(1, self.n_hidden,
                                        strides=self.n_hidden,
                                        activation="sigmoid"))
    return model

  def __call__(self) -> tf.Tensor:
    psi_re = self.model_re(self.all_confs)[:, :, 0]
    psi_im = self.model_im(self.all_confs)[:, :, 0]
    psi = tf.complex(psi_re, psi_im)
    return tf.concat([self.psi0, tf.transpose(psi)], axis=0)
