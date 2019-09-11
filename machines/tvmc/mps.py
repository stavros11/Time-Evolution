import numpy as np
from machines import mps
from utils import mps as utils

class SmallMPSStepMachine(mps.SmallMPSMachine):
  """MPS machine for small systems - uses dense wavefunctions.

  Used for traditional t-VMC evolution.
  """

  def __init__(self, init_state: np.ndarray, d_bond: int, d_phys: int = 2):
    self.n_states = len(init_state)
    self.n_sites = int(np.log2(self.n_states))
    self.d_bond, self.d_phys = d_bond, d_phys
    self.name = "stepmpsD{}".format(d_bond)

    tensors = np.array(utils.dense_to_mps(init_state, d_bond))
    self.tensors = tensors.transpose([0, 2, 1, 3])
    self.dtype = self.tensors.dtype
    self.shape = self.tensors.shape

    self._dense = self._create_envs()

  def dense(self) -> np.ndarray:
    return self._dense.reshape((self.n_states,))

  def wavefunction(self, configs: np.ndarray) -> np.ndarray:
    # Configs should be in {-1, 1} convention
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    return self._dense[configs_sl]

  def gradient(self, configs: np.ndarray)-> np.ndarray:
    # TODO: Redefine gradient in SmallMPSMachine class to avoid
    # code repetition here
    # Configs should be in {-1, 1} convention
    configs_t = (configs < 0).astype(configs.dtype).T
    n_samples = len(configs)
    srng = np.arange(n_samples)

    grads = np.zeros((n_samples,) + self.shape, dtype=self.dtype)

    right_slicer = tuple(configs_t[1:])
    grads[srng, 0, configs_t[0]] = self.right[-1][right_slicer].swapaxes(-2, -1)
    for i in range(1, self.n_sites - 1):
      left_slicer = tuple(configs_t[:i])
      left = self.left[i - 1][left_slicer]

      right_slicer = tuple(configs_t[i + 1:])
      right = self.right[self.n_sites - i - 2][right_slicer]

      grads[srng, i, configs_t[i]] = np.einsum("bmi,bjm->bij", left, right)

    left_slicer = tuple(configs_t[:-1])
    grads[srng, -1, configs_t[-1]] = self.left[-1][left_slicer].swapaxes(-2, -1)

    dense_slicer = tuple(configs_t) + len(self.shape) * (np.newaxis,)
    return grads / self._dense[dense_slicer]

  def update(self, to_add: np.ndarray) -> np.ndarray:
    if to_add.shape != self.shape:
      to_add = to_add.reshape(self.shape)

    self.tensors += to_add
    self._dense = self._create_envs()