"""C++ samplers test."""

import os
import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

sampler = ctypes.CDLL(os.path.join(os.getcwd(), "spacevmclib.so"))
sampler.run.argtypes = ([ndpointer(np.complex128, flags="C_CONTIGUOUS")] + 6 * [ctypes.c_int] +
                        [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")])
sampler.run.restype = None
print(sampler.run)