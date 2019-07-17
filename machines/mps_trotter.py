"""Evolution using the MPO form of the Trotterized evolution operator."""

import numpy as np
import scipy
import utils
from machines import mps_utils


def apply_prodop(op, mps):
  """Applies a product operator to an MPS.

  Product operator means a product of "one-qubit" gates.

  Args:
    op: Array that represents the product operator with shape (N, d, d).
    mps: MPS tensors to apply the operator to with shape (N, d, D, D).

  Returns:
    The resulting MPS after applying `op` with shape (N, d, D, D).
  """
  return np.einsum("ist,itlr->islr", op, mps)


def apply_mpo(op, mps):
  """Applies a Matrix Product Operator (MPO) to an MPS.

  Args:
    op: Array that represents the MPS with shape (N, Do, d, d, Do).
    mps: MPS tensors to apply the operator to with shape (N, d, D, D).

  Returns:
    The resulting MPS after applying `op` with shape (N, d, Dn, Dn).
      Here Dn = D * Do.
  """
  shape = list(mps.shape)
  shape[-2] *= op.shape[1]
  shape[-1] *= op.shape[-1]
  res = np.einsum("iLstR,itlr->isLlRr", op, mps)
  return res.reshape(shape)


def split_two_qubit_gate(u12, n_sites, d_bond):
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


def apply_two_qubit_product(ops, mps, even=True):
  """Applies a product of two qubit operations to an MPS.

  Works only for even number of sites in the MPS.

  Args:
    ops: The two qubit operator to apply as (N/2, d, d, d, d) array.
      The order of legs is (up1, up2, down1, down2).
    mps: The MPS to apply the operator to as (N, d, D, D) array.
    even: If true it starts from the first (index 0) site of the MPS,
      otherwise it starts at index 1 and respects PBC.

  Returns:
    The result of application as (N/2, d, d, D, D) array.
  """
  assert len(mps) % 2 == 0
  assert len(ops) == len(mps) // 2

  if even:
    mps = (mps[::2], mps[1::2])
  else:
    mps = (mps[1::2], np.stack((mps[2::2], mps[0][np.newaxis])))

  return np.einsum("iabcd,iclm,idmr->iablr", ops, mps[0], mps[1])


def truncate_bond_dimension(mps, d_bond):
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


def split_double_mps(dmps, even=True):
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
  mat = dmps.swapaxes((2, 3)).reshape((half_sites, mat_d, mat_d))
  # SVD
  u, s, v = np.linalg.svd(mat)
  # Truncate middle SVD dimension to D instead of d*D and reshape
  u = u[:, :, :d_bond].reshape((half_sites, d_phys, d_bond, d_bond))
  # Combine S and V
  ident = np.eye(d_bond, dtype=dtype)
  v = np.einsum("im,lm,imr->ilr", s[:, :d_bond], ident, v[:, :d_bond])
  v = v.reshape((half_sites, d_bond, d_phys, d_bond)).swapaxes((1, 2))

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



class TFIMBase:

  def __init__(self, psi0, d_bond):
    self.dtype = psi0.dtype
    self.d_bond = d_bond
    self.pauli = utils.Pauli(dtype=self.dtype)

    self.tensors = [mps_utils.dense_to_mps(psi0, d_bond, d_phys=2)]
    self.tensors[0] = self.tensors[0].swapaxes(1, 2)
    self.n_sites = len(self.tensors[0])

  def evolution_step():
    raise NotImplementedError

  def evolve(self, time_steps):
    for i in range(time_steps):
      self.tensors.append(self.evolution_step(self.tensors[-1]))
    return np.array(self.tensors)

  def dense_evolution(self, time_steps):
    if len(self.tensors) == 1:
      self.tensors = self.evolve(time_steps)
    assert len(self.tensors) == time_steps + 1
    return mps_utils.mps_to_dense(self.tensors.swapaxes(0, 1))


class TFIMTrotter(TFIMBase):

  def __init__(self, psi0, d_bond, dt, h=1.0):
    super().__init__(psi0, d_bond)
    self.x_op = self._construct_x_op(h, dt)
    self.zz_op = self._construct_zz_op(dt)

  def _construct_x_op(self, h, dt):
    cos, sin = np.cos(h * dt), np.sin(h * dt)
    exp_x = np.array([[cos, 1j * sin], [1j * sin, cos]], dtype=self.dtype)
    return np.stack(self.n_sites * [exp_x])

  def _construct_zz_op(self, dt):
    exp = np.exp(1j * dt)
    u12 = np.diag([exp, exp.conj(), exp.conj(), exp]).astype(self.dtype)
    return split_two_qubit_gate(u12, self.n_sites, d_bond=2)

  def evolution_step(self, mps0):
    zz_evolved = apply_mpo(self.zz_op, mps0)
    zz_trunc = truncate_bond_dimension(zz_evolved, self.d_bond)
    return apply_prodop(self.x_op, zz_trunc)


class TFIMFull(TFIMBase):

  def __init__(self, psi0, d_bond, dt, h=1.0):
    super().__init__(psi0, d_bond)
    pauli = utils.Pauli(self.dtype)
    ham12 = -np.kron(pauli.Z, pauli.Z)
    ham12 += -h * (np.kron(pauli.I, pauli.X) + np.kron(pauli.X, pauli.I))
    u12 = scipy.linalg.expm(-1j * dt * ham12)
    self.op = split_two_qubit_gate(u12, self.n_sites, d_bond=4)

  def evolution_step(self, mps0):
    evolved = apply_mpo(self.op, mps0)
    return truncate_bond_dimension(evolved, self.d_bond)
