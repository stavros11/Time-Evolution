"""Helper methods for Autodiff MPS optimization using contractions."""

def apply_mpo(op, mps):
  """Applies a Matrix Product Operator (MPO) to an MPS.

  Args:
    op: Array that represents the MPS with shape (N, Do, d, d, Do).
    mps: MPS tensors to apply the operator to with shape (..., N, d, D, D).

  Returns:
    The resulting MPS after applying `op` with shape (N, d, Dn, Dn).
      Here Dn = D * Do.
  """
  shape = list(mps.shape)
  shape[-2] *= op.shape[1]
  shape[-1] *= op.shape[-1]
  res = np.einsum("iLstR,itlr->isLlRr", op, mps)
  return res.reshape(shape)