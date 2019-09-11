"""/machines/t-VMC folder contains machines to be used with t-VMC."""
import numpy as np
from machines import base

class BaseStepMachine(base.BaseMachine):
  """Base machine to use with traditional t-VMC.

  Note that no machine inherits this class. It is only used for documentation
  purposes.

  Most methods are the same as above with the following diferences:
    `dense` returns shape (2^N).
    `wavefunction` takes only `configs` (not `times`) and
      returns shape (Ns,).
    `gradient` takes only `configs` (not `times`).
  """

  def __init__(self, n_sites: int):
    # __init__ should define the following attributes
    self.n_sites = n_sites
    self.dtype = None # Type of the variational parameters
    self.shape = None # Shape of the variational parameters
    self.name = None # Name (str) of the machine for saving purposes

  def update(self, s_matrix: np.ndarray, f_vector: np.ndarray):
    """Evolves variational parameters one time step.

    Uses Runge-Kutta integration.
    """
    raise NotImplementedError