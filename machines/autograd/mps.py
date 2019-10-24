import numpy as np
import tensorflow as tf
from machines.autograd import base
from utils.mps import mps as utils
from typing import Tuple


class SmallMPSModel(base.BaseAutoGrad):

  def __init__(self, **kwargs):
    init_state = kwargs["init_state"]
    super(SmallMPSModel, self).__init__(**kwargs)
    self.name = "smallmps_autograd"
    self.d_phys = 2
    self.d_bond = 4
    self.n_states = self.d_phys**self.n_sites

    tensors = np.array(
        self.time_steps * [utils.dense_to_mps(init_state, self.d_bond)])
    # make shape (n_sites, time_steps, d_phys, d_bond, d_bond)
    tensors = tensors.transpose([1, 0, 3, 2, 4])
    self.tensors_re = self.add_variable(tensors.real)
    self.tensors_im = self.add_variable(tensors.imag)

  def _add_complex_variable_with_noise(self, value, std=1e-3
                                       ) -> Tuple[tf.Tensor, tf.Tensor]:
    noise = np.random.normal(0.0, std, size=value.shape)
    re = self.add_variable(value + noise)
    noise = np.random.normal(0.0, std, size=value.shape)
    im = self.add_variable(value + noise)
    return re, im

  def ctensor(self, site: int):
    return tf.complex(self.tensors_re[site], self.tensors_im[site])

  def contract_sequentially(self):
    """Works for PBC.

    Returns:
      Dense wavefunction with shape (time_steps, n_states).
    """
    prod = self.ctensor(0)
    for i in range(1, self.n_sites):
      prod = tf.einsum("taij,tbjk->tabik", prod, self.ctensor(i))
      shape = list(prod.shape)
      prod = tf.reshape(prod, (shape[0], shape[1] * shape[2], shape[3], shape[4]))
    return tf.linalg.trace(prod)

  def forward(self):
    psi = self.contract_sequentially()
    return tf.concat([self.init_state, psi], axis=0)