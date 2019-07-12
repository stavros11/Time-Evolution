"""Evolution using the MPO form of the Trotterized evolution operator."""

import numpy as np
from machines import mps_utils
import utils


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


class TFIMTrotterMPS:

  def __init__(self, psi0, d_bond, dt, h=1.0):
    self.dtype = psi0.dtype
    self.d_bond = d_bond
    self.pauli = utils.Pauli(dtype=self.dtype)

    self.tensors = [mps_utils.dense_to_mps(psi0, d_bond, d_phys=2)]
    self.tensors[0] = self.tensors[0].swapaxes(1, 2)
    self.n_sites = len(self.tensors[0])

    self.x_op = self._construct_x_op(h, dt)
    self.zz_op = self._construct_zz_op(dt)

  def _construct_x_op(self, h, dt):
    cos, sin = np.cos(h * dt), np.sin(h * dt)
    exp_x = np.array([[cos, 1j * sin], [1j * sin, cos]], dtype=self.dtype)
    return np.stack(self.n_sites * [exp_x])

  def _construct_zz_op(self, dt):
    assert self.n_sites % 2 == 0
    exp = np.exp(1j * dt)
    u12 = np.diag([exp, exp.conj(), exp.conj(), exp]).astype(self.dtype)
    u12 = u12.reshape(4 * (2,)).transpose([0, 2, 1, 3])

    u, s, v = np.linalg.svd(u12.reshape((4, 4)))
    v = np.diag(s[:2]).dot(v[:2])

    u = u[:, :2].reshape((2, 2, 2)) # (d, d, D)
    v = v.reshape((2, 2, 2)) # (D, d, d)

    even = np.einsum("lum,mdr->ludr", v, u)
    odd = np.einsum("umr,lmd->ludr", u, v)
    return np.stack((self.n_sites // 2) * [even, odd])

  def evolution_step(self, mps0):
    zz_evolved = apply_mpo(self.zz_op, mps0)
    zz_trunc = truncate_bond_dimension(zz_evolved, self.d_bond)
    return apply_prodop(self.x_op, zz_trunc)

  def evolve(self, time_steps):
    for i in range(time_steps):
      self.tensors.append(self.evolution_step(self.tensors[-1]))
    return np.array(self.tensors)

  def dense_evolution(self, time_steps):
    if len(self.tensors) == 1:
      self.tensors = self.evolve(time_steps)
    assert len(self.tensors) == time_steps + 1
    return mps_utils.mps_to_dense(self.tensors.swapaxes(0, 1))
