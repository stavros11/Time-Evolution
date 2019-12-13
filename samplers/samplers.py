import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
from typing import Tuple


class Base:

  def __init__(self, n_sites: int, time_steps: int,
               n_samples: int, n_corr: int, n_burn: int):
    self.n_samples = n_samples
    self.n_corr = n_corr
    self.n_burn = n_burn

    self.n_sites = n_sites
    self.n_states = 2**n_sites
    self.time_steps = time_steps

    self.configs = np.zeros([n_samples, n_sites], dtype=np.int32)
    self.times = np.zeros(n_samples, dtype=np.int32)

  def __call__(self, full_psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


class SpinTime(Base):

  def __init__(self, **kwargs):
    super(SpinTime, self).__init__(**kwargs)

    self.cpp = ctypes.CDLL(os.path.join(os.getcwd(), "samplers", "qtvmclib.so"))
    self.cpp.run.argtypes = ([ndpointer(np.complex128, flags="C_CONTIGUOUS")] +
                             6 * [ctypes.c_int] +
                            [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                             ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")])
    self.cpp.run.restype = None

  def __call__(self, full_psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    self.cpp.run(full_psi, self.n_sites, self.time_steps + 1,
                 self.n_states, self.n_samples, self.n_corr, self.n_burn,
                 self.configs, self.times)
    return self.configs, self.times


class SpinOnly(Base):

  def __init__(self, **kwargs):
    super(SpinOnly, self).__init__(**kwargs)
    self.times = np.repeat(
        np.arange(self.time_steps + 1), self.n_samples).astype(self.times.dtype)

    self.cpp = ctypes.CDLL(os.path.join(os.getcwd(), "samplers", "spacevmclib.so"))
    self.cpp.run.argtypes = ([ndpointer(np.complex128, flags="C_CONTIGUOUS")] +
                             6 * [ctypes.c_int] +
                            [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")])
    self.cpp.run.restype = None

  def __call__(self, full_psi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    self.cpp.run(full_psi, self.n_sites, self.time_steps + 1,
                 self.n_states, self.n_samples, self.n_corr, self.n_burn,
                 self.configs)
    return self.configs, self.times