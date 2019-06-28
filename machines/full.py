"""Full Wavefunction Machine to be optimized using sampling."""

import numpy as np
from machines import base


class FullWavefunctionMachine(base.BaseMachine):

  def __init__(self, init_state, time_steps):
    self.n_sites = int(np.log2(len(init_state)))
    self.time_steps = time_steps
    # Initialize state
    self.psi = np.array((time_steps + 1) * [init_state])
    self.dtype = self.psi.dtype
    self.shape = self.psi[1:].shape

    self.bin_to_dec = 2**np.arange(0, self.n_sites)

  def set_parameters(self, psi):
    assert psi.shape == self.psi.shape
    self.psi = np.copy(psi)
    self.dtype = self.psi.dtype

  def dense(self):
    return self.psi

  def wavefunction(self, configs, times):
    configs_dec = (configs < 0).dot(self.bin_to_dec)

    psi_before = self.psi[np.clip(times - 1, 0, self.time_steps), configs_dec]
    psi_now = self.psi[times, configs_dec]
    psi_after = self.psi[np.clip(times+1, 0, self.time_steps), configs_dec]

    return np.stack((psi_before, psi_now, psi_after))

  def gradient(self, configs, times):
    n_samples = len(configs)
    configs_dec = (configs < 0).dot(self.bin_to_dec)

    grads = np.zeros((n_samples, self.psi.shape[-1]), dtype=self.dtype)
    grads[np.arange(n_samples), configs_dec] = 1.0 / self.psi[times, configs_dec]

    return grads

  def update(self, to_add):
    self.psi[1:] += to_add