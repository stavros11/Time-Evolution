"""Utilities for saving training data."""
import os
import h5py
import numpy as np
from typing import Dict, List


def save_histories(data_dir: str, filename: str,
                   history: Dict[str, List[float]]):
  """Saves history dictionary in an .h5 file in the given dir."""
  full_path = os.path.join(data_dir, "histories", filename)
  file = h5py.File("{}.h5".format(full_path), "w")
  for k in history.keys():
    file[k] = history[k]
  file.close()


def save_dense_wavefunction(data_dir: str, filename: str, full_psi: np.ndarray):
  """Saves a dense wavefunction to an .npy file."""
  full_path = os.path.join(data_dir, "final_dense", filename)
  np.save("{}.npy".format(full_path), full_psi)