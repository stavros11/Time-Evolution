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
    self._create_variables(init_state)

  def _create_variables(self, init_state: np.ndarray):
    tensors = np.array(
        self.time_steps * [utils.dense_to_mps(init_state, self.d_bond)])
    # make shape (time_steps, n_sites, d_phys, d_bond, d_bond)
    tensors = tensors.transpose([0, 1, 3, 2, 4])
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
    return tf.complex(self.tensors_re[:, site], self.tensors_im[:, site])

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


class SmallMPSProductPropModel(SmallMPSModel):

  def __init__(self, **kwargs):
    super(SmallMPSProductPropModel, self).__init__(**kwargs)
    self.name = "smallmps_prodprop"

  def _create_variables(self, init_state: np.ndarray):
    n = self.d_phys * self.d_bond**2
    prop = np.array(self.n_sites * [np.eye(n)])
    noise = np.random.normal(0.0, 1e-3, size=prop.shape)
    self.propagator_re = self.add_variable(prop + noise)
    noise = np.random.normal(0.0, 1e-3, size=prop.shape)
    self.propagator_im = self.add_variable(noise)

    init_tensors = utils.dense_to_mps(init_state, self.d_bond)
    init_tensors = init_tensors.swapaxes(1, 2)
    init_tensors = init_tensors.reshape((self.n_sites, n, 1))
    self.init_ctensor = tf.convert_to_tensor(
        init_tensors, dtype=self.output_type)

  def _calculate_ctensors(self):
    ctensors = [self.init_ctensor]
    propagator = tf.complex(self.propagator_re, self.propagator_im)
    for _ in range(self.time_steps):
      ctensors.append(tf.matmul(propagator, ctensors[-1]))

    ctensor = tf.stack(ctensors[1:])
    shape = (self.time_steps, self.n_sites, self.d_phys, self.d_bond, self.d_bond)
    return tf.reshape(ctensor, shape)

  def ctensor(self, site: int):
    return self.ctensors[:, site]

  def forward(self):
    self.ctensors = self._calculate_ctensors()
    return super(SmallMPSProductPropModel, self).forward()