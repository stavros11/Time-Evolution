"""Utilities for Matrix Product States."""
import numpy as np


def mps_to_dense(mps: np.ndarray) -> np.ndarray:
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


def dense_to_mps(state: np.ndarray, d_bond: int, d_phys: int = 2) -> np.ndarray:
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


def svd_split(m: np.ndarray):
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


def truncate_bond_dimension(mps: np.ndarray, d_bond: int) -> np.ndarray:
  """Truncates the MPS bond dimension to a smaller one using SVD.

  Assumes periodic boundary conditions.

  Args:
    mps: MPS tensors to truncate with shape (N, d, D, D).
    d_bond: New bond dimension (Dn).

  Returns:
    Truncated MPS tensors with shape (N, d, Dn, Dn).
  """
  n_sites, d_phys, old_bond, old_bond2 = mps.shape
  assert old_bond == old_bond2
  assert old_bond >= d_bond
  # SVD
  mats = mps.reshape((n_sites, d_phys * old_bond, old_bond))
  u, s, v = np.linalg.svd(mats)
  # Truncate
  ident = np.eye(d_bond, dtype=v.dtype)
  u = u[:, :, :d_bond].reshape((n_sites, d_phys, old_bond, d_bond))
  # (n_sites, d_bond, old_bond)
  v = np.einsum("im,lm,imr->ilr", s[:, :d_bond], ident, v[:, :d_bond])

  new = np.einsum("ilm,ismr->islr", v[:-1], u[1:])
  boundary = np.einsum("lm,smr->slr", v[-1], u[0])
  return np.concatenate((boundary[np.newaxis], new), axis=0)


def split_double_mps(dmps: np.ndarray, even: bool = True) -> np.ndarray:
  """Splits a pairwise contracted MPS to a normal MPS using SVD.

  Pairwise contracted MPS typically results by applying two qubit operator
  products using `apply_two_qubit_product`.
  The bond dimension is kept constant.

  Args:
    dmps: The pairwise contracted MPS of shape (N//2, d, d, D, D).
    even: If true splitting starts from site 0, else from site 1.
      (see `apply_two_qubit_product` for more details).

  Returns:
    Normal MPS of shape (N, d, D, D).
  """
  dtype = dmps.dtype
  # Reshape dmps to matrix for SVD
  half_sites, d_phys, _, d_bond, _ = dmps.shape
  mat_d = d_phys * d_bond
  mat = dmps.swapaxes(2, 3).reshape((half_sites, mat_d, mat_d))
  # SVD
  u, s, v = np.linalg.svd(mat)
  # Truncate middle SVD dimension to D instead of d*D and reshape
  u = u[:, :, :d_bond].reshape((half_sites, d_phys, d_bond, d_bond))
  # Combine S and V
  ident = np.eye(d_bond, dtype=dtype)
  v = np.einsum("im,lm,imr->ilr", s[:, :d_bond], ident, v[:, :d_bond])
  v = v.reshape((half_sites, d_bond, d_phys, d_bond)).swapaxes(1, 2)

  # Construct MPS
  mps = np.zeros((2 * half_sites, d_phys, d_bond, d_bond), dtype=dtype)
  if even:
    mps[::2] = u
    mps[1::2] = v
  else:
    mps[1::2] = u
    mps[2::2] = v[:-1]
    mps[0] = v[-1]

  return mps