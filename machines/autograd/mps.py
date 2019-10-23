import numpy as np
import tensorflow as tf
from machines.autograd import base


class SmallMPSModel(base.BaseAutoGrad):

  def __init__(self, **kwargs):
    init_state = kwargs["init_state"]
    super(SmallMPSModel, self).__init__(**kwargs)
    self.name = "fullwv_autograd"

    # Initialize full wavefunction by repeating the initial state
    init_value = np.array((self.time_steps + 1) * [init_state]).ravel()
    self.psi_re = self.add_variable(init_value.real)
    self.psi_im = self.add_variable(init_value.imag)

  def forward(self) -> tf.Tensor:
    psi = tf.complex(self.psi_re, self.psi_im)
    psi = tf.reshape(psi, self.dense_shape)
    return tf.concat([self.init_state, psi[1:]], axis=0)