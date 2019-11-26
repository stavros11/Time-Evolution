"""MPS machine for sampling."""

import numpy as np
from machines import base
from utils.mps import mps as utils
from typing import Tuple


class SmallMPS(base.BaseMachine):
  """MPS machine for small systems - uses dense wavefunctions."""

  @classmethod
  def create(cls, init_state: np.ndarray, time_steps: int, d_bond: int,
             d_phys: int = 2, learning_rate: float = 1e-3):
    n_states = len(init_state)
    n_sites = int(np.log2(n_states))
    name = "mpsD{}".format(d_bond)

    # Initialize tensors
    tensors = np.array((time_steps + 1) *
                       [utils.dense_to_mps(init_state, d_bond)])
    tensors = tensors.transpose([0, 1, 3, 2, 4])

    # Create machine
    machine =  cls(name, n_sites, tensors, learning_rate)
    machine.d_bond, machine.d_phys = d_bond, d_phys
    machine._dense = machine._create_envs()
    return machine

  @property
  def dense_tensor(self) -> np.ndarray:
    if self._dense is None:
      self._dense = self._create_envs()
    return self._dense

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
    return grads / self.dense_tensor[dense_slicer]

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