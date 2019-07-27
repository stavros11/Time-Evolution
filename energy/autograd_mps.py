"""Graphs for autograd MPS optimization using contractions."""

import tensorflow as tf


def apply_mpo(op, mps):
  """Applies a Matrix Product Operator (MPO) to an MPS.

  Args:
    op: Array that represents the MPO with shape (N, Do, d, d, Do).
    mps: MPS tensors to apply the operator to with shape (..., N, d, D, D).

  Returns:
    The resulting MPS after applying `op` with shape (N, d, Dn, Dn).
      Here Dn = D * Do.
  """
  shape = list(mps.shape)
  shape[-2] *= op.shape[1]
  shape[-1] *= op.shape[-1]
  res = tf.einsum("iLstR,...itlr->...isLlRr", op, mps)
  return tf.reshape(res, shape)


def pairwise_reduction(tensor):
  """Performs a sequence of matrix multiplication in pairs.

  Reduction axis must be -3.

  Args:
    tensor: Sequence of matrices with shape (..., N, D, D).

  Returns:
    Final matrices after reducing with shape (..., N).
  """
  size = int(tensor.shape[-3])
  # Idea from this implementation is from jemisjoky/TorchMPS
  while size > 1:
    half_size = size // 2
    nice_size = 2 * half_size
    leftover = tensor[..., nice_size:, :, :]
    tensor = tf.matmul(tensor[..., 0:nice_size:2, :, :],
                       tensor[..., 1:nice_size:2, :, :])
    tensor = tf.concat([tensor, leftover], axis=-3)
    size = half_size + int(size % 2 == 1)
  return tensor[..., 0, :, :]


def mps_overlap(mps1, mps2):
  """Calculates overlap <m1|m2> between of two MPSs.

  Contracts all physical indices first and then calls `pairwise_reduction`.
  The method takes the complex conjugate of the `mps1`.

  Args:
    mps1: First MPS tensors with shape (..., N, d, D1, D1).
    mps2: Second MPS tensors with shape (..., N, d, D2, D2).

  Returns:
    Overlap with shape (...,).
  """
  shape, shape2 = list(mps1.shape), list(mps2.shape)
  shape[-2] *= shape2[-2]
  shape[-1] *= shape2[-1]
  shape.pop(-3)

  prod = tf.einsum("...sLR,...slr->...LlRr", tf.conj(mps1), mps2)
  prod = pairwise_reduction(tf.reshape(prod, shape))
  return tf.linalg.trace(prod)


def clock_energy(mps, ham, dt):
  """Calculates clock EV <Psi|C|Psi> for a given MPS.

  Note that if |Psi> is not normalized we have to divide with <Psi|Psi>.
  We do not do the division here for flexibility. We may want to use this
  method with Lagrange multipliers instead of normalizing.

  Args:
    mps: MPS tensors with shape (T+1, N, d, D, D).
    ham: MPO representation of the Hamiltonian with shape (N, Do, d, d, Do).
    dt: (float) Size of time step.

  Returns:
    clock: <C> = <Psi|C|Psi> as a scalar.
    energy: Energy <H> for each time step with shape (T+1,).
    norms: Norm <psi|psi> at each time step with shape (T,).
      Note that the Clock-space norm <Psi|Psi> is sum(norms).
  """
  norms = mps_overlap(mps, mps)
  t1_t = mps_overlap(mps[1:], mps[:-1])

  term0 = tf.reduce_sum(2 * norms[:-1] - t1_t - tf.conj(t1_t))
  term0 += norms[-1] - norms[0]

  ham_mps = apply_mpo(ham, mps)
  energy = mps_overlap(mps, ham_mps)
  energy2 = mps_overlap(ham_mps, ham_mps)

  t1_ham_t = mps_overlap(mps[1:], ham_mps[:-1])
  term1 = tf.reduce_sum(t1_ham_t - tf.conj(t1_ham_t))

  term2 = tf.reduce_sum(energy2[1:-1])
  term2 += 0.5 * (energy2[0] + energy2[-1])

  clock = term0 + 1j * dt * term1 + dt * dt * term2

  return clock, energy, norms
