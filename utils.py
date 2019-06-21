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


def ev_local(state, op):
    """Expectation value of local operator.
    
    Args:
      state: Full state vector of shape (2**n_sites,)
      op: 2x2 matrix of local operator.
    """
    op_full = np.zeros((len(state), len(state)), dtype=op.dtype)
    identity = np.eye(2, dtype=op.dtype)
    n_sites = int(np.log2(len(state)))
    for site in range(n_sites):
      op_full += kron_list(site * [identity] + [op] + 
                           (n_sites - 1 - site) * [identity])
    return (np.conj(state) * op_full.dot(state)).sum() / n_sites
  

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


def tfim_exact_evolution(n_sites, t_final, time_steps, h0=1.0, h=0.5, 
                         dtype=np.complex128):
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
  Hinit = tfim_hamiltonian(n_sites, h=h0)
  Hevolve = tfim_hamiltonian(n_sites, h=h)
  pauli = Pauli(dtype=dtype)

  gs0 = la.eigh(Hinit)[1][:, 0]
  Udt = la.expm(-1j * dt * Hevolve)

  rtype = rtype_from_ctype(dtype)
  sigma_x = np.zeros(time_steps + 1, dtype=rtype)
  energy = np.zeros(time_steps + 1, dtype=rtype)

  state = [np.copy(gs0)]
  sigma_x[0] = ev_local(state[0], pauli.X).real
  energy[0] = (np.conj(state[0]) * Hevolve.dot(state[0])).sum().real
  for i in range(1, time_steps + 1):
    state.append(Udt.dot(state[i-1]))
    sigma_x[i] = ev_local(state[i], pauli.X).real
    energy[i] = (np.conj(state[i]) * Hevolve.dot(state[i])).sum().real

  observables = {"energy": energy, "X": sigma_x}  
  return np.array(state), observables


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
