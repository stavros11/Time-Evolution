"""Full Wavefunction Machine to be optimized using sampling."""

import numpy as np
from machines import base
from typing import Union


class FullWavefunction(base.BaseMachine):

  def __init__(self, init_state: np.ndarray, time_steps: int):
    self.n_states = len(init_state)
    self.n_sites = int(np.log2(self.n_states))
    self.time_steps = time_steps
    # Initialize state
    self.psi = np.array((time_steps + 1) * [init_state])
    self.psi = self.psi.reshape((time_steps + 1,) + self.n_sites * (2,))
    self.dtype = self.psi.dtype
    self.shape = self.psi[1:].shape
    self.name = "fullwv"

  def set_parameters(self, psi: np.ndarray):
    assert psi.shape == self.psi.shape
    self.psi = np.copy(psi)
    self.dtype = self.psi.dtype

  @property
  def dense(self) -> np.ndarray:
    return self.psi.reshape((self.time_steps + 1, self.n_states))

  def wavefunction(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    # Configs should be in {-1, 1}
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    times_before = np.clip(times - 1, 0, self.time_steps)
    times_after = np.clip(times + 1, 0, self.time_steps)

    psi_before = self.psi[(times_before,) + configs_sl]
    psi_now = self.psi[(times,) + configs_sl]
    psi_after = self.psi[(times_after,) + configs_sl]

    return np.stack((psi_before, psi_now, psi_after))

  def gradient(self, configs: np.ndarray, times: Union[int, np.ndarray]
               ) -> np.ndarray:
    # Configs should be in {-1, 1}
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    n_samples = len(configs)

    grads = np.zeros((n_samples,) + self.shape[1:], dtype=self.dtype)

    if isinstance(times, int):
      grads[(np.arange(n_samples),) + configs_sl] = (1.0 /
            self.psi[times][configs_sl])
    else:
      grads[(np.arange(n_samples),) + configs_sl] = (1.0 /
            self.psi[(times,) + configs_sl])
    return grads

  def update(self, to_add: np.ndarray):
    self.psi[1:] += to_add

  def update_time_step(self, new: np.ndarray, time_step: np.ndarray,
                       replace: bool = True):
    if replace:
      self.psi[time_step] = new.reshape(self.psi.shape[1:])
    else:
      self.psi[time_step] += new.reshape(self.psi.shape[1:])


class FullWavefunctionNormalized(FullWavefunction):

  def __init__(self, init_state: np.ndarray, time_steps: int):
    super().__init__(init_state, time_steps)
    self.name = "fullwvnorm"
    self.axes_to_sum = tuple(range(1, self.n_sites + 1))
    self.norm_slicer = (slice(None),) + self.n_sites * (np.newaxis,)

  def update(self, to_add: np.ndarray):
    self.psi[1:] += to_add
    norms = (np.abs(self.psi[1:])**2).sum(axis=self.axes_to_sum)
    self.psi[1:] *= 1.0 / np.sqrt(norms)[self.norm_slicer]