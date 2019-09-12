"""Utilities for applying operators to MPS."""
import numpy as np


def apply_prodop(op: np.ndarray, mps: np.ndarray) -> np.ndarray:
  """Applies a product operator to an MPS.

  Product operator means a product of "one-qubit" gates.

  Args:
    op: Array that represents the product operator with shape (N, d, d).
    mps: MPS tensors to apply the operator to with shape (N, d, D, D).

  Returns:
    The resulting MPS after applying `op` with shape (N, d, D, D).
  """
  return np.einsum("ist,itlr->islr", op, mps)


def apply_mpo(op: np.ndarray, mps: np.ndarray) -> np.ndarray:
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


def apply_two_qubit_product(ops: np.ndarray, mps: np.ndarray,
                            even: bool = True) -> np.ndarray:
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
    mps = (mps[1::2], np.concatenate((mps[2::2], mps[0][np.newaxis]), axis=0))

  return np.einsum("iabcd,iclm,idmr->iablr", ops, mps[0], mps[1])