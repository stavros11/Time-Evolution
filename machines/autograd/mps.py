import numpy as np
import tensorflow as tf
from machines.autograd import base
from typing import Tuple


class SmallMPSModel(base.BaseAutoGrad):

  def __init__(self, **kwargs):
    super(SmallMPSModel, self).__init__(**kwargs)
    self.name = "smallmps_autograd"
    self.d_phys = 2
    self.d_bond = 4
    self.n_states = self.d_phys**self.n_sites

    boundary_tensor = np.zeros((self.time_steps, self.d_phys, self.d_bond))
    boundary_tensor[:, 0, 0], boundary_tensor[:, 1, 0] = 1.0, 1.0
    middle_tensor = np.array(self.time_steps * self.d_phys * [np.eye(self.d_bond)])
    middle_tensor = middle_tensor.reshape(
        (self.time_steps, self.d_phys, self.d_bond, self.d_bond))

    self.tensors = [self._add_complex_variable_with_noise(boundary_tensor)]
    for _ in range(1, self.n_sites - 1):
      self.tensors.append(self._add_complex_variable_with_noise(middle_tensor))
    self.tensors.append(self._add_complex_variable_with_noise(boundary_tensor))

  def _add_complex_variable_with_noise(self, value, std=1e-3
                                       ) -> Tuple[tf.Tensor, tf.Tensor]:
    noise = np.random.normal(0.0, std, size=value.shape)
    re = self.add_variable(value + noise)
    noise = np.random.normal(0.0, std, size=value.shape)
    im = self.add_variable(value + noise)
    return re, im

  def forward(self) -> tf.Tensor:
    re, im = self.tensors[0]
    prod = tf.complex(re, im)
    for re, im in self.tensors[1:-1]:
      prod = tf.einsum("taj,tbjk->tabk", prod, tf.complex(re, im))
      shape = list(prod.shape)
      prod = tf.reshape(prod, (shape[0], shape[1] * shape[2], shape[3]))
    re, im = self.tensors[-1]
    prod = tf.einsum("taj,tbj->tab", prod, tf.complex(re, im))
    psi = tf.reshape(prod, (self.time_steps, self.n_states))
    return tf.concat([self.init_state, psi], axis=0)