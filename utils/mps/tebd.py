"""Evolution of MPS using the Trotterized evolution operator or TEBD."""
import numpy as np
import scipy
from utils import misc
from utils.mps import mps as mps_utils
from utils.mps import mpo as mpo_utils
from utils.mps import applyop


class EvolutionBase:

  def __init__(self, psi0: np.array, d_bond: int):
    self.dtype = psi0.dtype
    self.d_bond = d_bond
    self.pauli = misc.Pauli(dtype=self.dtype)

    self.tensors = [mps_utils.dense_to_mps(psi0, d_bond, d_phys=2)]
    self.tensors[0] = self.tensors[0].swapaxes(1, 2)
    self.n_sites = len(self.tensors[0])

  def evolution_step():
    raise NotImplementedError

  def evolve(self, time_steps: int) -> np.array:
    for i in range(time_steps):
      self.tensors.append(self.evolution_step(self.tensors[-1]))
    return np.array(self.tensors)

  def dense_evolution(self, time_steps: int) -> np.array:
    if len(self.tensors) == 1:
      self.tensors = self.evolve(time_steps)
    assert len(self.tensors) == time_steps + 1
    return mps_utils.mps_to_dense(self.tensors.swapaxes(0, 1))


class TFIMTrotter(EvolutionBase):
  """A self-made approach for MPS evolution of TFIM using MPOs."""

  def __init__(self, psi0: np.array, d_bond: int, dt: float, h: float = 1.0):
    super().__init__(psi0, d_bond)
    self.x_op = self._construct_x_op(h, dt)
    self.zz_op = self._construct_zz_op(dt)

  def _construct_x_op(self, h: float, dt: float) -> np.array:
    cos, sin = np.cos(h * dt), np.sin(h * dt)
    exp_x = np.array([[cos, 1j * sin], [1j * sin, cos]], dtype=self.dtype)
    return np.stack(self.n_sites * [exp_x])

  def _construct_zz_op(self, dt: float) -> np.array:
    exp = np.exp(1j * dt)
    u12 = np.diag([exp, exp.conj(), exp.conj(), exp]).astype(self.dtype)
    return mpo_utils.split_two_qubit_gate(u12, self.n_sites, d_bond=2)

  def evolution_step(self, mps0: np.array) -> np.array:
    zz_evolved = mpo_utils.apply_mpo(self.zz_op, mps0)
    zz_trunc = mps_utils.truncate_bond_dimension(zz_evolved, self.d_bond)
    return applyop.apply_prodop(self.x_op, zz_trunc)


class TFIM_TEBD(EvolutionBase):
  """Standard TEBD evolution for TFIM."""

  def __init__(self, psi0: np.array, d_bond: int, dt: float, h: float = 1.0):
    super().__init__(psi0, d_bond)
    pauli = misc.Pauli(self.dtype)
    ham12 = -np.kron(pauli.Z, pauli.Z)
    ham12 += -h * np.kron(pauli.X, pauli.I)

    exp = -1j * dt * ham12
    self.ops_even = self._construct_two_qubit_ops(exp / 2.0, self.n_sites)
    self.ops_odd = self._construct_two_qubit_ops(exp, self.n_sites)

  @staticmethod
  def _construct_two_qubit_ops(exponent: np.array, n_sites: int) -> np.array:
    u = scipy.linalg.expm(exponent)
    u = u.reshape(4 * (2,))
    return np.stack((n_sites // 2) * [u], axis=0)

  def _evolve_and_split(self, mps: np.array, even: bool = True) -> np.array:
    ops = [self.ops_odd, self.ops_even][even]
    dmps = applyop.apply_two_qubit_product(ops, mps, even=even)
    return mps_utils.split_double_mps(dmps, even=even)

  def evolution_step(self, mps0: np.array) -> np.array:
    mps = np.copy(mps0)
    for e in [True, False, True]:
      mps = self._evolve_and_split(mps, even=e)
    return mps