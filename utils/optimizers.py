"""Hard-coded optimizers that work with complex numbers."""
import numpy as np


class BaseOptimizer:

  def __init__(self, shape, dtype):
    self.shape = shape
    self.dtype = dtype

  def __call__(self, gradient, epoch):
    raise NotImplementedError


class AdamComplex(BaseOptimizer):
  """Adam optimizer for complex variable."""

  def __init__(self, shape, dtype, beta1=0.9, beta2=0.999, alpha=1e-3,
               epsilon=1e-8):
    super(AdamComplex, self).__init__(shape, dtype)
    self.m = np.zeros(shape, dtype=dtype)
    self.v = np.zeros(shape, dtype=dtype)
    self.beta1, self.beta2 = beta1, beta2
    self.alpha, self.eps = alpha, epsilon

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