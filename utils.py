import numpy as np
import scipy.linalg as la


def random_normal_complex(shape, loc=0.0, std=1.0, dtype=np.complex128):
  """Normally distributed complex number."""
  re = np.random.normal(loc, std, size=shape).astype(np.float64)
  im = np.random.normal(loc, std, size=shape).astype(np.float64)
  return (re + 1j * im).astype(dtype)


class Pauli():
  """Pauli matrices as np arrays."""

  def __init__(self, dtype=np.complex128):
    self.dtype = dtype
    self.I = np.eye(2).astype(dtype)
    self.X = np.array([[0, 1], [1, 0]], dtype=dtype)
    self.Y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    self.Z = np.array([[1, 0], [0, -1]], dtype=dtype)


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


def rtype_from_ctype(ctype):
  """Find float type from complex type"""
  n_complex = int(str(ctype).split("complex")[-1].split("'")[0])
  n_real = n_complex // 2
  return getattr(np, "float{}".format(n_real))


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
  pauli = Pauli(dtype=dtype)
  identities = [pauli.I for i in range(n_sites)]
  ham = np.zeros([2**n_sites, 2**n_sites], dtype=dtype)
  for i in range(n_sites - 1):
    matlist = identities[:]
    matlist[i], matlist[i + 1] = pauli.Z, pauli.Z
    ham += kron_list(matlist)
  if pbc:
    matlist = identities[:]
    matlist[0], matlist[-1] = pauli.Z, pauli.Z
    ham += kron_list(matlist)

  # Add field terms
  ham_f = np.zeros_like(ham)
  for i in range(n_sites - 1):
    ham_f += kron_list(identities[:i] + [pauli.X] + identities[i+1:])
  ham_f += kron_list(identities[:-1] + [pauli.X])

  return - ham - h * ham_f


def tfim_mpo(n_sites, h=1.0, dtype=np.complex128):
  """Constructs the Hamiltonian of periodic TFIM as a MPO.

  Args:
    n_sites: Number of sites.
    h: Transverse field strength.
    dtype: np type of the arrays.

  Returns:
    Periodic TFIM Hamiltonian as a MPO of shape (n_sites, 5, 2, 2, 5).
  """
  pauli = Pauli(dtype=dtype)
  assert n_sites > 4

  left_vector = np.zeros([5, 2, 2, 5], dtype=dtype)
  for i, p in enumerate([pauli.Z, h * pauli.X, pauli.I]):
    left_vector[0, :, :, i] = np.copy(p)

  right_vector = np.zeros([5, 2, 2, 5], dtype=dtype)
  for i, p in enumerate([-pauli.I, -h * pauli.X, -pauli.Z]):
    right_vector[i, :, :, 0] = np.copy(p)


  left_matrix = np.zeros([5, 2, 2, 5], dtype=dtype)
  left_matrix[0, :, :, 0] = pauli.Z
  left_matrix[0, :, :, 1] = pauli.I
  left_matrix[1, :, :, 0] = pauli.I
  left_matrix[2, :, :, 2] = pauli.Z
  left_matrix[2, :, :, 3] = h * pauli.X
  left_matrix[2, :, :, 4] = pauli.I

  right_matrix = np.zeros([5, 2, 2, 5], dtype=dtype)
  right_matrix[0, :, :, 0] = pauli.I
  right_matrix[1, :, :, 2] = pauli.I
  right_matrix[2, :, :, 0] = pauli.Z
  right_matrix[3, :, :, 0] = pauli.I
  right_matrix[4, :, :, 0] = h * pauli.X
  right_matrix[4, :, :, 1] = pauli.I
  right_matrix[4, :, :, 2] = pauli.Z

  middle_matrix = np.zeros([5, 2, 2, 5], dtype=dtype)
  middle_matrix[0, :, :, 0] = pauli.I
  middle_matrix[1, :, :, 1] = pauli.I
  middle_matrix[2, :, :, 0] = pauli.Z
  middle_matrix[3, :, :, 0] = pauli.I
  middle_matrix[4, :, :, 2] = pauli.Z
  middle_matrix[4, :, :, 3] = h * pauli.X
  middle_matrix[4, :, :, 4] = pauli.I

  return np.stack([left_vector, left_matrix] +
                  (n_sites - 4) * [middle_matrix] +
                  [right_matrix, right_vector])


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
  pauli = Pauli(dtype=dtype)

  if init_state is None:
    if h0 is None:
      h0 = 1.0
    Hinit = tfim_hamiltonian(n_sites, h=h0)
    init_state = la.eigh(Hinit)[1][:, 0]
  else:
    init_state = init_state.astype(dtype)

  rtype = rtype_from_ctype(dtype)
  sigma_x = np.zeros(time_steps + 1, dtype=rtype)
  energy = np.zeros(time_steps + 1, dtype=rtype)

  state = [np.copy(init_state)]
  sigma_x[0] = ev_local(state[0], pauli.X).real
  energy[0] = (np.conj(state[0]) * Hevolve.dot(state[0])).sum().real
  for i in range(1, time_steps + 1):
    state.append(Udt.dot(state[i-1]))
    sigma_x[i] = ev_local(state[i], pauli.X).real
    energy[i] = (np.conj(state[i]) * Hevolve.dot(state[i])).sum().real

  observables = {"energy": energy, "X": sigma_x}
  return np.array(state), observables


def tfim_trotterized_evolution(u_list, psi0, time_steps):
  """Trotterized evolution of full wavefunction.

  Args:
    u_list: List of unitary operators for one step time propagation.
    psi0: Initial condition with shape (2**N,).
    time_steps: Number of times to apply the evolution unitary.

  Returns:
    state: Evolved state.
  """
  state = [np.copy(psi0)]
  for i in range(time_steps):
    state_temp = np.copy(state[-1])
    for u in u_list:
      state_temp = u.dot(state_temp)
    state.append(np.copy(state_temp))
  return np.array(state)


class AdamComplex():
  """Adam optimizer for complex variable."""

  def __init__(self, shape, dtype, beta1=0.9, beta2=0.999, alpha=1e-3, epsilon=1e-8):
    self.m = np.zeros(shape, dtype=dtype)
    self.v = np.zeros(shape, dtype=dtype)
    self.beta1, self.beta2 = beta1, beta2
    self.alpha, self.eps = alpha, epsilon

  def update(self, gradient, epoch):
    self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
    self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient.real**2 + 1j * gradient.imag**2)
    beta1t, beta2t = self.beta1**(epoch + 1), self.beta2**(epoch + 1)
    mhat = self.m / (1 - beta1t)
    vhat = self.v / (1 - beta2t)
    a_t = self.alpha * np.sqrt(1 - beta2t) / (1 - beta1t)
    return -a_t * (mhat.real / (np.sqrt(vhat.real) + self.eps) + 1j * mhat.imag / (np.sqrt(vhat.imag) + self.eps))
