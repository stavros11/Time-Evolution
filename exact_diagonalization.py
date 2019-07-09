"""Methods for constructing the Clock Hamiltonian for ED."""

import numpy as np
from scipy import sparse


def construct_sparse_clock(ham, dt, time_steps, init_penalty=0.0, psi0=None):
  """Constructs Clock Hamiltonian as a sparse array.

  Args:
    ham: Real space evolution Hamiltonian.
    dt: Time step size (float).
    time_steps: Number of time steps (denoted by T in equations).
      Note that time Hilbert space has dimension = time_steps + 1.

  Returns:
    clock: The Clock Hamiltonian in the space (space\otimes time) as a sparse
      array.
  """
  dtype = ham.dtype

  # Time Hilbert space projectors
  proj_0 = np.zeros((time_steps + 1, time_steps + 1), dtype=dtype)
  proj_T = np.zeros((time_steps + 1, time_steps + 1), dtype=dtype)
  proj_0[0, 0], proj_T[-1, -1] = 1, 1

  # Time Hilbert space sums
  q_sum = np.diag(np.ones(time_steps), k=1).astype(dtype)
  p_sum = np.diag(np.ones(time_steps), k=1).astype(dtype)
  q_sum = q_sum + q_sum.T
  p_sum = 1j * (p_sum.T - p_sum)

  # Identities of time and space Hilbert spaces
  id_t = np.eye(time_steps + 1, dtype=dtype)
  id_h = np.eye(ham.shape[0], dtype=dtype)

  clock = sparse.kron(2 * id_t - q_sum - proj_0 - proj_T, id_h)
  clock += dt * sparse.kron(p_sum, ham)
  clock += dt * dt * sparse.kron(id_t - 0.5 * (proj_0 + proj_T), ham.dot(ham))

  if init_penalty > 0.0:
    if psi0 is None:
      raise ValueError("Positive penaly coefficient without specified initial"
                       " condition.")

    proj_init = psi0[:, np.newaxis].dot(psi0[np.newaxis])
    clock += sparse.kron(init_penalty * proj_0, id_h - proj_init)

  return clock


def solve_evolution_system(clock, psi0, solver):
  """Finds evolution for a given initial state by solving the Clock system.

  Notice the 1 / np.sqrt(T + 1) for the initial condition that normalizes the
  Clock state.

  Args:
    clock: Clock Hamiltonian (without penalty term), as constructed by
      `construct_sparse_clock`.
    psi0: Initial condition wavefunction as an array of (2**n_sites,).
    solver: A scipy.sparse.linalg solver method (such as gmres) to solve
      the linear system.

  Returns:
    Whatever the solver returns, usually (solution, info).
  """
  dim_space = len(psi0)
  norm = np.sqrt(clock.shape[0] // dim_space)
  lhs = clock[dim_space:, dim_space:]
  rhs = - clock[dim_space:, :dim_space].dot(psi0) / norm
  return solver(lhs, rhs)


