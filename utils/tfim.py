"""Utilities that are relevant to the TFIM model."""
import numpy as np
import scipy.linalg as la
from utils import misc
from utils import calc


def tfim_hamiltonian(n_sites, h=1.0, pbc=True, dtype=np.complex128):
  """Creates nearest neighbor TFIM Hamiltonian matrix.

  Args:
    n_sites: Number of sites.
    h: Transverse field strength.
    pbc: If True, periodic boundary conditions are assumed.
    dtype: np type of the arrays.

  Returns:
    TFIM Hamiltonian matrix of shape (2**n_sites, 2**n_sites).
  """
  pauli = misc.Pauli(dtype=dtype)
  identities = [pauli.I for i in range(n_sites)]
  ham = np.zeros([2**n_sites, 2**n_sites], dtype=dtype)
  for i in range(n_sites - 1):
    matlist = identities[:]
    matlist[i], matlist[i + 1] = pauli.Z, pauli.Z
    ham += calc.kron_list(matlist)
  if pbc:
    matlist = identities[:]
    matlist[0], matlist[-1] = pauli.Z, pauli.Z
    ham += calc.kron_list(matlist)

  # Add field terms
  ham_f = np.zeros_like(ham)
  for i in range(n_sites - 1):
    ham_f += calc.kron_list(identities[:i] + [pauli.X] + identities[i+1:])
  ham_f += calc.kron_list(identities[:-1] + [pauli.X])

  return - ham - h * ham_f


def tfim_exact_evolution(n_sites, t_final, time_steps, h0=None, h=0.5,
                         init_state=None, dtype=np.complex128):
  """Exact unitary evolution of TFIM using full propagator matrix.

  Args:
    n_sites: Number of sites.
    t_final: Time evolution duration.
    time_steps: Number of time steps to evolve (initial point does not count.)
    h0: Initial field. The initial condition is the corresponding ground state.
    h: Evolution Hamiltonian field. We are evolving a quench.
    dtype: Complex array types.

  Returns:
    state: Evolved state in every time step with shape (M + 1, 2**n_sites)
      where M = time_steps.
    observables: Dictionary with energy and sigma_x for every time step.
  """
  dt = t_final / time_steps
  Hevolve = tfim_hamiltonian(n_sites, h=h)
  Udt = la.expm(-1j * dt * Hevolve)
  pauli = misc.Pauli(dtype=dtype)

  if init_state is None:
    if h0 is None:
      h0 = 1.0
    Hinit = tfim_hamiltonian(n_sites, h=h0)
    init_state = la.eigh(Hinit)[1][:, 0]
  else:
    if (np.abs(init_state)**2).sum() != 1.0:
      raise ValueError("Given initial state is not normalized.")
    init_state = init_state.astype(dtype)

  rtype = misc.rtype_from_ctype(dtype)
  sigma_x = np.zeros(time_steps + 1, dtype=rtype)
  energy = np.zeros(time_steps + 1, dtype=rtype)

  state = [np.copy(init_state)]
  sigma_x[0] = calc.ev_local(state[0], pauli.X).real
  energy[0] = (np.conj(state[0]) * Hevolve.dot(state[0])).sum().real
  for i in range(1, time_steps + 1):
    state.append(Udt.dot(state[i-1]))
    sigma_x[i] = calc.ev_local(state[i], pauli.X).real
    energy[i] = (np.conj(state[i]) * Hevolve.dot(state[i])).sum().real

  observables = {"energy": energy, "X": sigma_x}
  return np.array(state), observables
