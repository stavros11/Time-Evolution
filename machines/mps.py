"""MPS machine for sampling."""

import numpy as np
from machines import base
from optimization import deterministic
from utils.mps import mps as utils
from typing import Callable, Tuple


class SmallMPS(base.BaseMachine):
  """MPS machine for small systems - uses dense wavefunctions."""

  def __init__(self, init_state: np.ndarray, time_steps: int, d_bond: int,
               d_phys: int = 2):
    self.n_states = len(init_state)
    self.n_sites = int(np.log2(self.n_states))
    self.time_steps = time_steps
    self.d_bond, self.d_phys = d_bond, d_phys
    self.name = "mpsD{}".format(d_bond)

    tensors = np.array((time_steps + 1) *
                       [utils.dense_to_mps(init_state, d_bond)])
    self.tensors = tensors.transpose([0, 1, 3, 2, 4])
    self.dtype = self.tensors.dtype
    self.shape = self.tensors[1:].shape

    self._dense = self._create_envs()

  @property
  def dense(self) -> np.ndarray:
    return self._dense.reshape((self.time_steps + 1, self.n_states))

  @property
  def deterministic_gradient_func(self) -> Callable:
    return deterministic.sampling_gradient

  def _vectorized_svd_split(self, m: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Splits multiple MPS tensors simultaneously.

    Args:
      m: MPS tensors with shape (..., d, D, D)

    Returns:
      ml: Left tensor after splits with shape (..., d, D, D)
      mr: Right tensor after split with shape (..., d, D, D)
    """
    batch_shape = m.shape[:-3]
    d, Dl, Dr = m.shape[-3:]
    assert Dl == self.d_bond
    assert Dr == self.d_bond

    u, s, v = np.linalg.svd(m.reshape(batch_shape + (d * Dl, Dr)),
                            full_matrices=False)
    sm = np.zeros(batch_shape + (self.d_bond, self.d_bond), dtype=m.dtype)
    slicer = len(batch_shape) * (slice(None),)
    sm[slicer + 2 * (range(self.d_bond),)] = s

    u = u.reshape(batch_shape + (d, self.d_bond, self.d_bond))
    v = (sm[slicer + (slice(None), slice(None), np.newaxis)] *
         v[slicer + (np.newaxis, slice(None), slice(None))]).sum(axis=-2)
    return u, v

  def _to_canonical_form(self):
    for i in range(self.n_sites - 1):
      self.tensors[:, i], v = self._vectorized_svd_split(self.tensors[:, i])
      self.tensors[:, i + 1] = np.einsum("tlm,tsmr->tslr", v,
                  self.tensors[:, i + 1])

  def _calculate_dense(self) -> np.ndarray:
    """Calculates the dense form of MPS.

    NOT USED (only for tests).

    Note that currently tests are broken because _calculate_dense returns
    the dense in (M+1, 2**N), while _create_envs returns the dense in
    (M+1, 2, 2, ..., 2) shape. We are currently following the latter convention.
    """
    n = self.n_sites
    d = self.d_phys
    tensors = np.copy(self.tensors).swapaxes(0, 1)
    while n > 1:
      n = n // 2
      d *= d
      tensors = np.einsum("italm,itbmr->itablr", tensors[::2], tensors[1::2])
      tensors = tensors.reshape(
          (n, self.time_steps + 1, d, self.d_bond, self.d_bond))
    return np.trace(tensors[0], axis1=-2, axis2=-1)

  def _expr(self, s1: str, s2: str) -> str:
    """Returns einsum expression for _create_envs."""
    return "...{}xy,...{}yz->...{}{}xz".format(s1, s2, s1, s2)

  SYMBOLS = "abcdefghijklmnopqrstuvw"

  def _create_envs(self) -> np.ndarray:
    """Creates left and right environments.

    Also calculates dense.

    self.left is a list with all the left contractions in dense form.
    Therefore the shapes are
    [(d, D, D), (d, d, D, D), (d, d, d, D, D), ..., (N-1)*(d,) + (D, D)]
    and similarly for self.right.
    The time index is included as the first axis in the above shapes every time.
    """
    tensor = lambda i: self.tensors[..., i, :, :, :]

    self.left = [np.copy(tensor(0))]
    self.right = [np.copy(tensor(-1))]
    for i in range(1, self.n_sites - 1):
      expr = self._expr(self.SYMBOLS[:i], self.SYMBOLS[i])
      self.left.append(np.einsum(expr, self.left[-1], tensor(i)))

      expr = self._expr(self.SYMBOLS[i], self.SYMBOLS[:i])
      self.right.append(np.einsum(expr, tensor(self.n_sites - i - 1),
                                  self.right[-1]))

    expr = self._expr(self.SYMBOLS[:self.n_sites - 1],
                      self.SYMBOLS[self.n_sites - 1])
    dense = np.einsum(expr, self.left[-1], tensor(-1))
    return np.trace(dense, axis1=-2, axis2=-1)

  def wavefunction(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    # Configs should be in {-1, 1} convention
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    times_before = np.clip(times - 1, 0, self.time_steps)
    times_after = np.clip(times + 1, 0, self.time_steps)

    psi_before = self._dense[(times_before,) + configs_sl]
    psi_now = self._dense[(times,) + configs_sl]
    psi_after = self._dense[(times_after,) + configs_sl]

    return np.stack((psi_before, psi_now, psi_after))

  def gradient(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    # Configs should be in {-1, 1} convention
    configs_t = (configs < 0).astype(configs.dtype).T
    n_samples = len(configs)
    srng = np.arange(n_samples)

    grads = np.zeros((n_samples,) + self.shape[1:], dtype=self.dtype)

    right_slicer = (times,) + tuple(configs_t[1:])
    grads[srng, 0, configs_t[0]] = self.right[-1][right_slicer].swapaxes(-2, -1)
    for i in range(1, self.n_sites - 1):
      left_slicer = (times,) + tuple(configs_t[:i])
      left = self.left[i - 1][left_slicer]

      right_slicer = (times,) + tuple(configs_t[i + 1:])
      right = self.right[self.n_sites - i - 2][right_slicer]

      grads[srng, i, configs_t[i]] = np.einsum("bmi,bjm->bij", left, right)

    left_slicer = (times,) + tuple(configs_t[:-1])
    grads[srng, -1, configs_t[-1]] = self.left[-1][left_slicer].swapaxes(-2, -1)

    dense_slicer = (times,) + tuple(configs_t)
    dense_slicer += (len(self.shape) - 1) * (np.newaxis,)
    return grads / self._dense[dense_slicer]

  def update(self, to_add: np.ndarray) -> np.ndarray:
    self.tensors[1:] += to_add
    self._dense = self._create_envs()

  def update_time_step(self, new: np.ndarray, time_step: np.ndarray):
    # Assume that dense is a dense form and NOT an MPS in order to use
    # the `ExactGMResSweep` sweeper as it is for the full wavefunction
    new_mps = utils.dense_to_mps(new, self.d_bond)
    self.tensors[time_step] = new_mps.swapaxes(1, 2)
    # TODO: This call can be optimized as we can only recalculate the `envs`
    # of the time step that we updated, while `_create_envs()` recalculates
    # all environments as it was designed for global optimization
    self._dense = self._create_envs()


class SmallMPSNormalized(SmallMPS):
  """Normalizes the wavefunction by dividing every MPS tensor with the norm.

  Norm calculation is tractable only for small systems.
  """

  def __init__(self, init_state: np.ndarray, time_steps: int, d_bond: int,
               d_phys: int = 2):
    super().__init__(init_state, time_steps, d_bond, d_phys=2)
    self.name = "mpsD{}norm".format(d_bond)
    self.axes_to_sum = tuple(range(1, self.n_sites + 1))
    tensor_slicer = (slice(None),) + len(self.tensors.shape[1:]) * (np.newaxis,)
    self.dense_slicer = (slice(None),) + self.n_sites * (np.newaxis,)

    norms = np.sqrt((np.abs(self._dense)**2).sum(axis=self.axes_to_sum))
    self.tensors *= 1.0 / (norms[tensor_slicer])**(1 / self.n_sites)

  def update(self, to_add: np.ndarray) -> np.ndarray:
    self.tensors[1:] += to_add
    self._dense = self._create_envs()
    norms = np.sqrt((np.abs(self._dense)**2).sum(axis=self.axes_to_sum))
    self._dense *= 1.0 / norms[self.dense_slicer]


class SmallMPSCanonical(SmallMPS):
  """Normalizes the wavefunction by turning MPS to canonical form.

  DOES NOT WORK PROPERLY.
  """

  def __init__(self, init_state: np.ndarray, time_steps: int, d_bond: int,
               d_phys: int = 2):
    super().__init__(init_state, time_steps, d_bond, d_phys=2)
    self.name = "mpsD{}canonical".format(d_bond)
    self.canonical_mask = (self.tensors != 0.0).astype(self.dtype)
    self._to_canonical_form()
    self._dense = self._create_envs()

  def update(self, to_add: np.ndarray) -> np.ndarray:
    self.tensors[1:] += to_add
    self.tensors *= self.canonical_mask
    self._to_canonical_form()
    self._dense = self._create_envs()