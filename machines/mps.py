"""MPS machine for sampling."""

import numpy as np
from machines import base


class SmallMPSMachine(base.BaseMachine):
  """MPS machine for small systems - uses dense wavefunctions."""

  def __init__(self, init_state, time_steps, d_bond, d_phys=2):
    self.n_states = len(init_state)
    self.n_sites = int(np.log2(self.n_states))
    self.time_steps = time_steps
    self.d_bond, self.d_phys = d_bond, d_phys
    self.name = "mpsD{}".format(d_bond)

    tensors = np.array((time_steps + 1) * [self._dense_to_mps(init_state)])
    self.tensors = tensors.transpose([0, 1, 3, 2, 4])
    self.dtype = self.tensors.dtype
    self.shape = self.tensors[1:].shape

    self._dense = self._create_envs()
    self.bin_to_dec = 2**np.arange(self.n_sites)

  def _dense_to_mps(self, state):
    """Transforms a dense wavefunction to an approximate MPS form."""
    tensors = [state[np.newaxis, :, np.newaxis]]
    while len(tensors) < self.n_sites:
      tensors[-1], temp = self._svd_split(tensors[-1])
      tensors.append(temp)

    array = np.zeros([self.n_sites, self.d_bond, self.d_phys, self.d_bond],
                     dtype=state.dtype)
    for i in range(self.n_sites):
      array[i, :tensors[i].shape[0], :, :tensors[i].shape[-1]] = tensors[i][:self.d_bond, :, :self.d_bond]
    return array

  @staticmethod
  def _svd_split(m):
    """Splits an MPS tensor in two MPS tensors.

    Args:
      m: MPS tensor with shape (Dl, d, Dr)

    Returns:
      ml: Left tensor after split with shape (Dl, d', Dm)
      mr: Right tensor after split with shape (Dm, d', Dr)
      with d' = sqrt(d) and Dm = d' min(Dl, Dr)
    """
    Dl, d, Dr = m.shape
    u, s, v = np.linalg.svd(m.reshape(Dl * 2, Dr * d // 2))
    D_middle = min(u.shape[-1], v.shape[0])
    s = np.diag(s[:D_middle])

    u = u[:, :D_middle].reshape((Dl, 2, D_middle))
    sv = s.dot(v[:D_middle]).reshape((D_middle, d // 2, Dr))
    return u, sv

  def _vectorized_svd_split(self, m):
    """Splits multiple MPS tensors simultaneously.

    Args:
      m: MPS tensors with shape (..., d, D, D)

    Returns:
      ml: Left tensor after splits with shape (..., d, D, D)
      mr: Right tensor after split with shape (..., d, D, D)
    """
    batch_shape = m.shape[:-3]
    d, Dl, Dr = m.shape[-3:]
    assert Dl == self.d_bond
    assert Dr == self.d_bond

    u, s, v = np.linalg.svd(m.reshape(batch_shape + (d * Dl, Dr)), full_matrices=False)
    sm = np.zeros(batch_shape + (self.d_bond, self.d_bond), dtype=m.dtype)
    slicer = len(batch_shape) * (slice(None),)
    sm[slicer + 2 * (range(self.d_bond),)] = s

    u = u.reshape(batch_shape + (d, self.d_bond, self.d_bond))
    v = (sm[slicer + (slice(None), slice(None), np.newaxis)] *
         v[slicer + (np.newaxis, slice(None), slice(None))]).sum(axis=-2)
    return u, v

  def _to_canonical_form(self):
    for i in range(self.n_sites - 1):
      self.tensors[:, i], v = self._vectorized_svd_split(self.tensors[:, i])
      self.tensors[:, i + 1] = np.einsum("tlm,tsmr->tslr", v, self.tensors[:, i + 1])

  def _calculate_dense(self):
    """Calculates the dense form of MPS.

    NOT USED (only for tests).

    Note that currently tests are broken because _calculate_dense returns
    the dense in (M+1, 2**N), while _create_envs returns the dense in
    (M+1, 2, 2, ..., 2) shape. We are currently following the latter convention.
    """
    n = self.n_sites
    d = self.d_phys
    tensors = np.copy(self.tensors).swapaxes(0, 1)
    while n > 1:
      n = n // 2
      d *= d
      tensors = np.einsum("italm,itbmr->itablr", tensors[::2], tensors[1::2])
      tensors = tensors.reshape(
          (n, self.time_steps + 1, d, self.d_bond, self.d_bond))
    return np.trace(tensors[0], axis1=-2, axis2=-1)

  def _expr(self, s1, s2):
    """Returns einsum expression for _create_envs."""
    return "t{}xy,t{}yz->t{}{}xz".format(s1, s2, s1, s2)

  SYMBOLS = "abcdefghijklmnopqrstuvw"

  def _create_envs(self):
    """Creates left and right environments.

    Also calculates dense.

    self.left is a list with all the left contractions in dense form.
    Therefore the shapes are
    [(d, D, D), (d, d, D, D), (d, d, d, D, D), ..., (N-1)*(d,) + (D, D)]
    and similarly for self.right.
    The time index is included as the first axis in the above shapes every time.
    """
    self.left = [np.copy(self.tensors[:, 0])]
    self.right = [np.copy(self.tensors[:, -1])]
    for i in range(1, self.n_sites - 1):
      expr = self._expr(self.SYMBOLS[:i], self.SYMBOLS[i])
      self.left.append(np.einsum(expr, self.left[-1], self.tensors[:, i]))

      expr = self._expr(self.SYMBOLS[i], self.SYMBOLS[:i])
      self.right.append(np.einsum(expr, self.tensors[:, self.n_sites - i - 1],
                                  self.right[-1]))

    expr = self._expr(self.SYMBOLS[:self.n_sites - 1],
                      self.SYMBOLS[self.n_sites - 1])
    dense = np.einsum(expr, self.left[-1], self.tensors[:, -1])
    return np.trace(dense, axis1=-2, axis2=-1)

  def dense(self):
    return self._dense.reshape((self.time_steps + 1, self.n_states))

  def wavefunction(self, configs, times):
    # Configs should be in {-1, 1} convention
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    times_before = np.clip(times - 1, 0, self.time_steps)
    times_after = np.clip(times + 1, 0, self.time_steps)

    psi_before = self._dense[(times_before,) + configs_sl]
    psi_now = self._dense[(times,) + configs_sl]
    psi_after = self._dense[(times_after,) + configs_sl]

    return np.stack((psi_before, psi_now, psi_after))

  def gradient(self, configs, times):
    # Configs should be in {-1, 1} convention
    configs_t = (configs < 0).astype(configs.dtype).T
    n_samples = len(configs)
    srng = np.arange(n_samples)

    grads = np.zeros((n_samples,) + self.shape[1:], dtype=self.dtype)

    right_slicer = (times,) + tuple(configs_t[1:])
    grads[srng, 0, configs_t[0]] = self.right[-1][right_slicer].swapaxes(-2, -1)
    for i in range(1, self.n_sites - 1):
      left_slicer = (times,) + tuple(configs_t[:i])
      left = self.left[i - 1][left_slicer]

      right_slicer = (times,) + tuple(configs_t[i + 1:])
      right = self.right[self.n_sites - i - 2][right_slicer]

      grads[srng, i, configs_t[i]] = np.einsum("bmi,bjm->bij", left, right)

    left_slicer = (times,) + tuple(configs_t[:-1])
    grads[srng, -1, configs_t[-1]] = self.left[-1][left_slicer].swapaxes(-2, -1)

    dense_slicer = (times,) + tuple(configs_t)
    dense_slicer += (len(self.shape) - 1) * (np.newaxis,)
    return grads / self._dense[dense_slicer]

  def update(self, to_add):
    self.tensors[1:] += to_add
    self._dense = self._create_envs()


class SmallMPSMachineNorm(SmallMPSMachine):

  def __init__(self, init_state, time_steps, d_bond, d_phys=2):
    super().__init__(init_state, time_steps, d_bond, d_phys=2)
    self.name = "mpsD{}norm".format(d_bond)
    self.axes_to_sum = tuple(range(1, self.n_sites + 1))
    self.norm_slicer = (slice(None),) + len(self.tensors.shape[1:]) * (np.newaxis,)

    norms = np.sqrt((np.abs(self._dense)**2).sum(axis=self.axes_to_sum))
    self.tensors *= 1.0 / (norms[self.norm_slicer])**(1 / self.n_sites)

  def update(self, to_add):
    self.tensors[1:] += to_add
    temp_dense = self._create_envs()
    norms = np.sqrt((np.abs(temp_dense[1:])**2).sum(axis=self.axes_to_sum))

    self.tensors[1:] *= 1.0 / (norms[self.norm_slicer])**(1 / self.n_sites)
    self._dense = self._create_envs()


class SmallMPSMachineCanonical(SmallMPSMachine):

  def __init__(self, init_state, time_steps, d_bond, d_phys=2):
    super().__init__(init_state, time_steps, d_bond, d_phys=2)
    self.name = "mpsD{}canonical".format(d_bond)
    self.canonical_mask = (self.tensors != 0.0).astype(self.dtype)
    self._to_canonical_form()
    self._dense = self._create_envs()

  def update(self, to_add):
    self.tensors[1:] += to_add
    self.tensors *= self.canonical_mask
    self._to_canonical_form()
    self._dense = self._create_envs()
