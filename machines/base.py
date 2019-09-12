"""Base machine to be used for sampling calculations.

All machines should inherit this.
Machines are used when optimizing with sampling.
"""
import numpy as np
from typing import Callable


class BaseMachine:
  """Base machine to use with Clock optimization."""

  def __init__(self, n_sites: int, time_steps: int):
    # Time steps do not include initial condition
    # __init__ should define the following attributes
    self.n_sites = n_sites
    self.time_steps = time_steps
    self.dtype = None # Type of the variational parameters
    self.shape = None # Shape of the variational parameters
    self.name = None # Name (str) of the machine for saving purposes

  @property
  def dense(self) -> np.ndarray:
    """Calculates the dense (full wavefunction) form of the machine.

    This is used because the C++ sampler currently only supports full
    wavefunctions to avoid much communication with Python.

    Returns:
      dense form of the wavefunction with shape (M+1, Ns)
    """
    raise NotImplementedError

  @property
  def deterministic_gradient_func(self) -> Callable:
    """Machine's deterministic gradient calculation function.

    This is used because not all machines are compatible with all gradient
    functions. For example MPS machines are not compatible with
    `deterministic.gradient` but instead should use
    `deterministic.sampling_gradient`.

    Returns:
      One of the functions defined in `optimization.determinstic`.
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

  def update(self, to_add: np.ndarray) -> np.ndarray:
    """Updates variational parameters.

    Args:
      to_add: Value to add to the variational parameters.
    """
    raise NotImplementedError