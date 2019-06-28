"""MPS machine for sampling."""

import numpy as np
from machines import base


class SmallMPSMachine(base.BaseMachine):
  """MPS machine for small systems - uses dense wavefunctions."""

  def __init__(self, init_state, time_steps, d_bond, d_phys=2):
    self.n_sites = int(np.log2(len(init_state)))
    self.time_steps = time_steps
    self.d_bond, self.d_phys = d_bond, d_phys

    tensors = np.array((time_steps + 1) * [self._dense_to_mps(init_state)])
    self.tensors = tensors.transpose([0, 1, 3, 2, 4])
    self.dtype = self.tensors.dtype
    self.shape = self.tensors[1:].shape

    self._dense = self._calculate_dense()

  def _dense_to_mps(self, state):
    """Transforms a dense wavefunction to an approximate MPS form."""
    tensors = [state[np.newaxis, :, np.newaxis]]
    while len(tensors) < self.n_sites:
      tensors = [m for t in tensors for m in self._svd_split(t)]

    array = np.zeros([self.n_sites, self.d_bond, self.d_phys, self.d_bond],
                     dtype=state.dtype)
    for i in range(self.n_sites):
      array[i, :tensors[i].shape[0], :, :tensors[i].shape[-1]] = tensors[i][:self.d_bond, :, :self.d_bond]
    return array

  def _svd_split(self, m):
    """Splits an MPS tensor in two MPS tensors.

    Args:
      m: MPS tensor with shape (Dl, d, Dr)

    Returns:
      ml: Left tensor after split with shape (Dl, d', Dm)
      mr: Right tensor after split with shape (Dm, d', Dr)
      with d' = sqrt(d) and Dm = d' min(Dl, Dr)
    """
    Dl, d, Dr = m.shape
    d_new = int(np.sqrt(d))
    u, s, v = np.linalg.svd(m.reshape(Dl * d_new, Dr * d_new))
    d_middle = min(u.shape[-1], v.shape[0])
    s = np.diag(s[:d_middle])

    u = u[:, :d_middle].reshape((Dl, d_new, d_middle))
    sv = s.dot(v[:d_middle]).reshape((d_middle, d_new, Dr))
    return u, sv

  def _sequential_dense(self, tensors):
    """To be used for gradients."""



  def _calculate_dense(self):
    """Calculates the dense form of MPS."""
    n = self.n_sites
    d = self.d_phys
    tensors = np.copy(self.tensors).swapaxes(0, 1)
    while n > 1:
      n = n // 2
      d *= d
      tensors = np.einsum("italm,itbmr->itablr", tensors[::2], tensors[1::2])
      tensors = tensors.reshape(
          (n, self.time_steps, d, self.d_bond, self.d_bond))
    return np.trace(tensors[0], axis1=-2, axis2=-1)

  def dense(self):
    return self._dense

  def wavefunction(self, configs, times):
    configs_dec = (configs < 0).dot(self.bin_to_dec)

    psi_before = self._dense[np.clip(times - 1, 0, self.time_steps), configs_dec]
    psi_now = self._dense[times, configs_dec]
    psi_after = self._dense[np.clip(times + 1, 0, self.time_steps), configs_dec]

    return np.stack((psi_before, psi_now, psi_after))

  def gradient(self, configs, times):


  def update(self, to_add):
    self.tensors += to_add
    self._dense = self._calculate_dense()