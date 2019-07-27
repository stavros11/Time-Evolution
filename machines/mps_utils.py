"""Various utilities for Matrix Product States."""

import numpy as np


def mps_to_dense(mps):
  """Transforms a MPS chain to a dense wavefunction.

  Args:
    mps: Array of MPS tensors with shape (N, ..., d, D, D).

  Returns:
    Dense wavefunction of shape (..., d**N,)
  """
  def reshape(x):
    shape = list(x.shape)
    shape[-3] *= shape.pop(-4)
    return x.reshape(shape)

  dense = reshape(np.einsum("...slm,...tmr->...stlr", mps[0], mps[1]))
  for m in mps[2:]:
    dense = reshape(np.einsum("...slm,...tmr->...stlr", dense, m))
  return np.trace(dense, axis1=-2, axis2=-1)


def mpo_to_dense(mpo, trace=True):
  """Transforms a MPO to a dense unitary operator.

  Args:
    mpo: Array of MPO tensors with shape (N, ..., D, d, d, D).
    trace: If True, it finishes the calculation by taking the trace,
      that is contracting the two dangling legs.
      If False, it sums all the elements of the final matrix. This is used
      when the first and last sites are vectors but are passed as diagonal
      matrices. In this case the sum over all elements is equivalent to the
      actual contraction.

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

  if trace:
    return np.trace(dense, axis1=-4, axis2=-1)
  else:
    return dense.sum(axis=(-4, -1))


def dense_to_mps(state, d_bond, d_phys=2):
  """Transforms a dense wavefunction to an approximate MPS form using SVD.

  Args:
    state: Dense state to transform with shape (d**N,)
    d_bond: Bond dimension (D) of the MPS to create.
      D is constant across the chain.
    d_phys: Physical dimension (d) of the MPS.

  Returns:
    Constructed MPS with shape (N, D, d, D).
  """
  tensors = [state[np.newaxis, :, np.newaxis]]
  if d_phys == 2:
    n_sites = int(np.log2(len(state)))
  else:
    n_sites = int(np.log(len(state)) / np.log(d_phys))

  while len(tensors) < n_sites:
    tensors[-1], temp = svd_split(tensors[-1])
    tensors.append(temp)

  array = np.zeros([n_sites, d_bond, d_phys, d_bond], dtype=state.dtype)
  for i in range(n_sites):
    s = tensors[i].shape
    array[i, :s[0], :, :s[-1]] = tensors[i][:d_bond, :, :d_bond]
  return array


def svd_split(m):
  """Splits an MPS tensor in two MPS tensors.

  Args:
    m: MPS tensor with shape (Dl, d, Dr)

  Returns:
    ml: Left tensor after split with shape (Dl, d', Dm)
    mr: Right tensor after split with shape (Dm, d', Dr)
      with d' = sqrt(d) and Dm = d' min(Dl, Dr)
  """
  Dl, d, Dr = m.shape
  u, s, v = np.linalg.svd(m.reshape(Dl * 2, Dr * d // 2))
  D_middle = min(u.shape[-1], v.shape[0])
  s = np.diag(s[:D_middle])

  u = u[:, :D_middle].reshape((Dl, 2, D_middle))
  sv = s.dot(v[:D_middle]).reshape((D_middle, d // 2, Dr))
  return u, sv