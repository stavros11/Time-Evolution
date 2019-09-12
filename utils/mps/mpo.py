"""Utilities for Matrix Product Operators."""
import numpy as np


def mpo_to_dense(mpo: np.ndarray) -> np.ndarray:
  """Transforms a MPO to a dense unitary operator.

  Args:
    mpo: Array of MPO tensors with shape (N, ..., D, d, d, D).

  Returns:
    Dense operator with shape (..., d**N, d**N)
  """
  def reshape(x):
    shape = list(x.shape)
    shape[-4] *= shape.pop(-5)
    shape[-2] *= shape.pop(-3)
    return x.reshape(shape)

  dense = reshape(np.einsum("lUDm,mudr->lUuDdr", mpo[0], mpo[1]))
  for m in mpo[2:]:
    dense = reshape(np.einsum("lUDm,mudr->lUuDdr", dense, m))

  return np.trace(dense, axis1=-4, axis2=-1)


def split_two_qubit_gate(u12: np.ndarray, n_sites: int,
                         d_bond: int) -> np.ndarray:
  """"Splits a two qubit gate to an MPO for n_sites.

  Args:
    u12: The two qubit gate as a (4, 4) array.
    n_sites: Number of sites for the MPO construction.
    d_bond: Bond dimension D of the MPO.

  Returns:
    The MPO that is equivalent to applying the two qubit gate to all nn
    pairs of the given sites as a (N, D, d, d, D) array.
  """
  assert n_sites % 2 == 0
  u12 = u12.reshape(4 * (2,)).transpose([0, 2, 1, 3])

  u, s, v = np.linalg.svd(u12.reshape((4, 4)))
  v = np.diag(s[:d_bond]).dot(v[:d_bond])

  u = u[:, :d_bond].reshape((2, 2, d_bond)) # (d, d, D)
  v = v.reshape((d_bond, 2, 2)) # (D, d, d)

  even = np.einsum("lum,mdr->ludr", v, u)
  odd = np.einsum("umr,lmd->ludr", u, v)
  return np.stack((n_sites // 2) * [even, odd])