"""Neural network that propagates the variational parameters."""

import numpy as np
import tensorflow as tf
from models import simple
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM


class MPSLSTM(simple.MPSModel):

  def __init__(self, init_state, time_steps, d_bond, d_phys=2,
               rtype=tf.float32, ctype=tf.complex64):
    self.n_sites = int(np.log2(len(init_state)))
    self.time_steps = time_steps
    self.d_bond, self.d_phys = d_bond, d_phys


    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)

    tensors = np.array(time_steps * [self._dense_to_mps(init_state)])

    self.mps_shape = tensors.shape
    shape = (time_steps, np.prod(self.mps_shape[1:]))

    tensors = tensors.reshape(shape)
    self.input_re = tf.cast(tensors.real[np.newaxis], dtype=rtype)
    self.input_im = tf.cast(tensors.imag[np.newaxis], dtype=rtype)

    self.lstm_re = Sequential([InputLayer(shape),
                               LSTM(shape[-1], return_sequences=True)])
    self.lstm_im = Sequential([InputLayer(shape),
                               LSTM(shape[-1], return_sequences=True)])

    self.vars = self.lstm_re.variables + self.lstm_im.variables

  def variational_wavefunction(self, training=False):
    output_re = self.lstm_re(self.input_re)[0]
    output_im = self.lstm_im(self.input_im)[0]
    tensors = tf.reshape(tf.complex(output_re, output_im), self.mps_shape)
    return self._contract_tensors(tf.transpose(tensors, [1, 0, 3, 2, 4]))