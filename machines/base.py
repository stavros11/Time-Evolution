"""Base machine to be used for sampling calculations.

All machines should inherit this.
Machines are used when optimizing with sampling.
"""
import numpy as np
from utils import optimizers
from typing import List, Optional, Tuple


class BaseMachine:
  """Base machine to use with Clock optimization."""

  def __init__(self, name: str, n_sites: int, tensors: np.ndarray,
               learning_rate: float = 1e-3):
    """Create machine object given the variational parameters.

    Args:
      name: Name of the machine used for logging and saving purposes.
      n_sites: Number of physical sites in the system.
      tensors: Variational parameters. Must have shape (time_steps + 1, ...).
      learning_rate: Learning rate of the optimizer.
    """
    # Time steps do not include initial condition
    self.name = name # Name (str) of the machine for saving purposes
    self.n_sites = n_sites
    self.time_steps = len(tensors) - 1
    self.n_states = 2**n_sites

    self.tensors = tensors
    # Optimizer to use for updating the variational parameters
    self.optimizer = optimizers.AdamComplex(self.shape, self.dtype,
                                            alpha=learning_rate)
    # Placeholder for dense wavefunction
    self._dense = None

  @property
  def dense_tensor(self) -> np.ndarray:
    """Returns the dense (full wavefunction) form of the machine.

    Returns:
      dense form of the wavefunction with shape (M+1, 2, 2, ..., 2).
    """
    if self._dense is None:
      raise NotImplementedError
    return self._dense

  def gradient(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculates gradient value on given samples.

    Args:
      configs: Spin configuration samples of shape (Ns, N)
      times: Time configuration samples of shape (Ns,)

    Returns:
      gradient of shape (Ns,) + variational parameters shape
    """
    raise NotImplementedError

  @property
  def shape(self) -> Tuple[int]:
    """Shape of the variational parameters."""
    return self.tensors[1:].shape

  @property
  def dtype(self):
    return self.tensors.dtype

  @property
  def dense(self) -> np.ndarray:
    """Returns the dense (full wavefunction) form of the machine.

    This is used because the C++ sampler currently only supports full
    wavefunctions to avoid much communication with Python.

    Returns:
      dense form of the wavefunction with shape (M+1, Ns)
    """
    shape = (self.time_steps + 1, self.n_states)
    return self.dense_tensor.reshape(shape)

  def set_parameters(self, tensors: np.ndarray,
                     time_steps: Optional[List[int]] = None):
    if time_steps is None:
      assert tensors.shape == self.tensors.shape
      self.tensors = np.copy(tensors)
    else:
      assert tensors.shape[1:] == self.tensors.shape[1:]
      assert len(time_steps) == len(tensors)
      self.tensors[time_steps] = np.copy(tensors)
    self._dense = None

  def wavefunction(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculates wavefunction value on given samples.

    For each time we have to calculate the wavefunction for the previous and
    next time step to calculate all energy terms!

    Args:
      configs: Spin configuration samples of shape (Ns, N)
      times: Time configuration samples of shape (Ns,)

    Returns:
      wavefunction of shape (3, Ns)
    """
    # Configs should be in {-1, 1} convention
    configs_sl = tuple((configs < 0).astype(configs.dtype).T)
    times_before = np.clip(times - 1, 0, self.time_steps)
    times_after = np.clip(times + 1, 0, self.time_steps)

    psi_before = self.dense_tensor[(times_before,) + configs_sl]
    psi_now = self.dense_tensor[(times,) + configs_sl]
    psi_after = self.dense_tensor[(times_after,) + configs_sl]
    return np.stack((psi_before, psi_now, psi_after))

  def update(self, grad: np.ndarray, epoch: int, update_zero: bool = False):
    """Updates variational parameters.

    Args:
      grad: Gradient to use for updating the variational parameters.
      epoch: Epoch number of optimization (needed for Adam optimizer).
    """
    if update_zero:
      if grad.shape != self.tensors.shape:
        grad = grad.reshape(self.tensors.shape)
      to_add = self.optimizer(grad, epoch)
      self.tensors += to_add

    else:
      if grad.shape != self.shape:
        grad = grad.reshape(self.shape)
      to_add = self.optimizer(grad, epoch)
      self.tensors[1:] += to_add

    self._dense = None

  @classmethod
  def subset(cls, time_steps: List[int], machine, update_zero: bool = False
             ) -> "BaseMachine":
    """Creates a subset machine by keeping specific time steps"""
    new_tensors = machine.tensors[time_steps]
    old_optimizer = machine.optimizer
    new_machine = cls(machine.name, machine.n_sites, new_tensors)
    if update_zero:
      new_shape = new_machine.tensors.shape
    else:
      new_shape = new_machine.shape
    new_machine.optimizer = old_optimizer.renew(new_shape, new_machine.dtype,
                                                old_optimizer)
    return new_machine