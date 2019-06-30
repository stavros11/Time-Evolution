"""Full Wavefunction Machine to be optimized using sampling."""

import numpy as np
from machines import base


class FullWavefunctionMachine(base.BaseMachine):

  def __init__(self, init_state, time_steps):
    self.n_states = len(init_state)
    self.n_sites = int(np.log2(self.n_states))
    self.time_steps = time_steps
    # Initialize state
    self.psi = np.array((time_steps + 1) * [init_state])
    self.psi = self.psi.reshape((time_steps + 1,) + self.n_sites * (2,))
    self.dtype = self.psi.dtype
    self.shape = self.psi[1:].shape

  def set_parameters(self, psi):
    assert psi.shape == self.psi.shape
    self.psi = np.copy(psi)
    self.dtype = self.psi.dtype

  def dense(self):
    return self.psi.reshape((self.time_steps + 1, self.n_states))

  def wavefunction(self, configs, times):
    # Configs should be in {-1, 1}
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    times_before = np.clip(times - 1, 0, self.time_steps)
    times_after = np.clip(times + 1, 0, self.time_steps)

    psi_before = self.psi[(times_before,) + configs_sl]
    psi_now = self.psi[(times,) + configs_sl]
    psi_after = self.psi[(times_after,) + configs_sl]

    return np.stack((psi_before, psi_now, psi_after))

  def gradient(self, configs, times):
    # Configs should be in {-1, 1}
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    n_samples = len(configs)

    grads = np.zeros((n_samples,) + self.shape[1:], dtype=self.dtype)
    grads[(np.arange(n_samples),) + configs_sl] = (1.0 /
          self.psi[(times,) + configs_sl])

    return grads

  def update(self, to_add):
    self.psi[1:] += to_add