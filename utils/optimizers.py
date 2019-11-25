"""Hard-coded optimizers that work with complex numbers."""
import numpy as np


class BaseOptimizer:

  def __init__(self):
    pass

  def __call__(self, gradient, epoch):
    self.params = None # dict placeholder
    raise NotImplementedError

  @classmethod
  def renew(cls, shape, dtype, optimizer):
    """Creates a new optimizer of the same type with an update shape.

    Useful when growing in time, where the shape of machines changes.
    """
    params = optimizer.params
    params["shape"] = shape
    params["dtype"] = dtype
    return cls(**params)


class AdamComplex(BaseOptimizer):
  """Adam optimizer for complex variable."""

  def __init__(self, shape, dtype, beta1=0.9, beta2=0.999, alpha=1e-3,
               epsilon=1e-8):
    self.m = np.zeros(shape, dtype=dtype)
    self.v = np.zeros(shape, dtype=dtype)
    self.beta1, self.beta2 = beta1, beta2
    self.alpha, self.eps = alpha, epsilon

    self.params = {"beta1": beta1, "beta2": beta2,
                   "alpha": alpha, "epsilon": epsilon}

  def __call__(self, gradient, epoch):
    self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
    comp_grad2 = gradient.real**2 + 1j * gradient.imag**2
    self.v = self.beta2 * self.v + (1 - self.beta2) * comp_grad2

    beta1t, beta2t = self.beta1**(epoch + 1), self.beta2**(epoch + 1)
    mhat = self.m / (1 - beta1t)
    vhat = self.v / (1 - beta2t)
    a_t = self.alpha * np.sqrt(1 - beta2t) / (1 - beta1t)

    re = mhat.real / (np.sqrt(vhat.real) + self.eps)
    im = mhat.imag / (np.sqrt(vhat.imag) + self.eps)
    return -a_t * (re + 1j * im)