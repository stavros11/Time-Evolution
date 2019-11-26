"""Full Wavefunction Machine to be optimized using sampling."""

import numpy as np
from machines import base


class FullWavefunction(base.BaseMachine):

  @classmethod
  def create(cls, init_state: np.ndarray, time_steps: int,
             learning_rate: float = 1e-3):
    n_states = len(init_state)
    n_sites = int(np.log2(n_states))

    # Initialize parameters by copying initial state
    tensors = np.array((time_steps + 1) * [init_state])
    tensors = tensors.reshape((time_steps + 1,) + n_sites * (2,))

    return cls("fullwv", n_sites, tensors, learning_rate)

  @property
  def dense_tensor(self) -> np.ndarray:
    return self.tensors

  def gradient(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    # Configs should be in {-1, 1}
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    n_samples = len(configs)

    grads = np.zeros((n_samples,) + self.shape[1:], dtype=self.dtype)
    grads[(np.arange(n_samples),) + configs_sl] = (1.0 /
          self.psi[(times,) + configs_sl])
    return grads