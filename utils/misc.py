"""Miscaleneous utilities."""
import numpy as np


def rtype_from_ctype(ctype):
  """Find float type from complex type"""
  n_complex = int(str(ctype).split("complex")[-1].split("'")[0])
  n_real = n_complex // 2
  return getattr(np, "float{}".format(n_real))


def random_normal_complex(shape, loc=0.0, std=1.0, dtype=np.complex128):
  """Returns normally distributed complex numbers."""
  re = np.random.normal(loc, std, size=shape).astype(np.float64)
  im = np.random.normal(loc, std, size=shape).astype(np.float64)
  return (re + 1j * im).astype(dtype)


class Pauli():
  """Pauli matrices as np arrays."""

  def __init__(self, dtype=np.complex128):
    self.dtype = dtype
    self.I = np.eye(2).astype(dtype)
    self.X = np.array([[0, 1], [1, 0]], dtype=dtype)
    self.Y = np.array([[0, -1j], [1j, 0]], dtype=dtype)
    self.Z = np.array([[1, 0], [0, -1]], dtype=dtype)