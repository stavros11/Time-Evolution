"""Base machine to be used for sampling calculations.

All machines should inherit this.
"""


class BaseMachine:

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