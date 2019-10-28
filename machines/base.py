"""Base machine to be used for sampling calculations.

All machines should inherit this.
Machines are used when optimizing with sampling.
"""
import numpy as np
from typing import Tuple


class BaseMachine:
  """Base machine to use with Clock optimization."""

  def __init__(self, name: str, n_sites: int, time_steps: int):
    # TODO: Support optimizer type - currently not useful as we only have
    # complex Adam implemented

    # Time steps do not include initial condition
    # __init__ should define the following attributes
    self.name = name # Name (str) of the machine for saving purposes
    self.n_sites = n_sites
    self.time_steps = time_steps
    self.dtype = None # Type of the variational parameters
    # Optimizer to use for updating the variational parameters
    self.optimizer = None

  @property
  def shape(self) -> Tuple[int]:
    """Shape of the variational parameters."""
    raise NotImplementedError

  @property
  def dense(self) -> np.ndarray:
    """Calculates the dense (full wavefunction) form of the machine.

    This is used because the C++ sampler currently only supports full
    wavefunctions to avoid much communication with Python.

    Returns:
      dense form of the wavefunction with shape (M+1, Ns)
    """
    raise NotImplementedError

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
    raise NotImplementedError

  def gradient(self, configs: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Calculates gradient value on given samples.

    Args:
      configs: Spin configuration samples of shape (Ns, N)
      times: Time configuration samples of shape (Ns,)

    Returns:
      gradient of shape (Ns,) + variational parameters shape
    """
    raise NotImplementedError

  def update(self, grad: np.ndarray, epoch: int):
    """Updates variational parameters.

    Args:
      grad: Gradient to use for updating the variational parameters.
      epoch: Epoch number of optimization (needed for Adam optimizer).
    """
    # TODO: Define this in `Base` so that you don't have to define it
    # again for all machines indepedently.
    raise NotImplementedError

  def add_time_step(self):
    """Adds an additional time step to the machine's parameters.

    Used when growing in time during optimization.
    """
    raise NotImplementedError