"""Base machine to be used for sampling calculations.

All machines should inherit this.
"""


class BaseMachine:
  """Base machine to use with Clock optimization."""

  def __init__(self, n_sites, time_steps):
    # Time steps do not include initial condition
    # __init__ should define the following attributes
    self.n_sites = n_sites
    self.time_steps = time_steps
    self.dtype = None # Type of the variational parameters
    self.shape = None # Shape of the variational parameters
    self.name = None # Name (str) of the machine for saving purposes

  def dense(self):
    """Calculates the dense (full wavefunction) form of the machine.

    This is used because the C++ sampler currently only supports full
    wavefunctions to avoid much communication with Python.

    Returns:
      dense form of the wavefunction with shape (M+1, Ns)
    """
    raise NotImplementedError

  def wavefunction(self, configs, times):
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

  def gradient(self, configs, times):
    """Calculates gradient value on given samples.

    Args:
      configs: Spin configuration samples of shape (Ns, N)
      times: Time configuration samples of shape (Ns,)

    Returns:
      gradient of shape (Ns,) + variational parameters shape
    """
    raise NotImplementedError

  def update(self, to_add):
    """Updates variational parameters.

    Args:
      to_add: Value to add to the variational parameters.
    """
    raise NotImplementedError


class BaseStepMachine(BaseMachine):
  """Base machine to use with traditional t-VMC.

  Note that no machine inherits this class. It is only used for documentation
  purposes.

  Most methods are the same as above with the following diferences:
    `dense` returns shape (2^N).
    `wavefunction` takes only `configs` (not `times`) and
      returns shape (Ns,).
    `gradient` takes only `configs` (not `times`).
  """

  def __init__(self, n_sites):
    # __init__ should define the following attributes
    self.n_sites = n_sites
    self.dtype = None # Type of the variational parameters
    self.shape = None # Shape of the variational parameters
    self.name = None # Name (str) of the machine for saving purposes

  def update(self, s_matrix, f_vector):
    """Evolves variational parameters one time step.

    Uses Runge-Kutta integration.
    """
    raise NotImplementedError