"""Utilities that calculate overlaps of states and EVs of operators."""
import numpy as np


def kron_list(matrices_list):
  """Kronecker product of list of matrices.

  Args:
    matrices_list: List of matrices as numpy arrays.

  Returns:
    Kronecker product of all the matrices in the list.
  """
  term = np.kron(matrices_list[0], matrices_list[1])
  if len(matrices_list) < 3:
    return term
  else:
    return kron_list([term] + matrices_list[2:])


def overlap(state1, state2, normalize_states=True):
  """Calculates overlap between two `clock` states.

  Args:
    state1: State of shape (M + 1, 2**N)
    state2: State of shape (M + 1, 2**N)
    normalize_states: If true it normalizes the states before the overlap
      calculation.

  Returns:
    Overlap as a real number.
  """
  prod = (state1.conj() * state2).sum()
  if normalize_states:
    norm1 = (np.abs(state1)**2).sum()
    norm2 = (np.abs(state2)**2).sum()
    return np.abs(prod)**2 / (norm1 * norm2)
  return np.abs(prod)**2


def time_overlap(state1, state2, normalize_states=True):
  """Calculates the overlap two state evolutions as a function of time.

  Args:
    state1: State of shape (M + 1, 2**N)
    state2: State of shape (M + 1, 2**N)
    normalize_states: If true it normalizes the states before the overlap
      calculation.

  Returns:
    Overlap with shape (M + 1,)
  """
  prod = (state1.conj() * state2).sum(axis=1)
  if normalize_states:
    norms1 = (np.abs(state1)**2).sum(axis=1)
    norms2 = (np.abs(state2)**2).sum(axis=1)
    return (np.abs(prod)**2 / (norms1 * norms2))
  return np.abs(prod)**2


def averaged_overlap(state1, state2, normalize_states=True):
  """Calculates averaged over time overlap between two state evolutions.

  Args:
    state1: State of shape (M + 1, 2**N)
    state2: State of shape (M + 1, 2**N)
    normalize_states: If true it normalizes the states before the overlap
      calculation.

  Returns:
    Overlap as a real number.
  """
  return time_overlap(state1, state2, normalize_states=normalize_states).mean()


def ev_local(state, op):
    """Expectation value of local operator.

    Args:
      state: Full state vector of shape (2**N,) or (M + 1, 2**N)
      op: 2x2 matrix of local operator.
    """
    n_states = state.shape[-1]
    op_full = np.zeros((n_states, n_states), dtype=op.dtype)
    identity = np.eye(2, dtype=op.dtype)
    n_sites = int(np.log2(n_states))
    for site in range(n_sites):
      op_full += kron_list(site * [identity] + [op] +
                           (n_sites - 1 - site) * [identity])
    if len(state.shape) < 2:
      return (np.conj(state) * op_full.dot(state)).sum() / n_sites
    op_state = op_full.dot(state.T).T
    return (np.conj(state) * op_state).sum(axis=1) / n_sites