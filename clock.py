"""Calculate Clock energy given a dense wavefunction."""
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple


def energy(psi_all: tf.Tensor, Ham: tf.Tensor, dt: float,
           Ham2: Optional[tf.Tensor] = None) -> tf.Tensor:
  """Calculates Clock's expectation value.

  Args:
    psi_all: Full wavefunction at each time step of shape (M+1, 2**N)
      where N is the number of sites and M+1 the number of time steps.
    Ham: Full (real space) Hamiltonian matrix of shape (2**N, 2**N).
    dt: Time step.
    Ham2: Square of spin Hamiltonian of shape (2**N, 2**N).
      This can optionally be given to avoid calculating it multiple times

  Returns:
    Heff_samples: Array of shape (M+1, 2**N) with the Hamiltonian terms
      before the reduce sum.
  """
  if Ham2 is None:
    Ham2 = tf.matmul(Ham, Ham)

  # H^0 term
  n_Heff_0 = psi_all[0] - psi_all[1]
  n_Heff_M = psi_all[-1] - psi_all[-2]
  n_Heff = 2 * psi_all[1:-1] - psi_all[:-2] - psi_all[2:]

  # H^1 term
  n_Heff_0 -= 1j * dt * tf.matmul(Ham, psi_all[1][:, tf.newaxis])[:, 0]
  n_Heff_M += 1j * dt * tf.matmul(Ham, psi_all[-2][:, tf.newaxis])[:, 0]
  n_Heff += 1j * dt * tf.matmul(psi_all[:-2] - psi_all[2:], Ham,
                                transpose_b=True)

  # H^2 term
  n_Heff2_all = tf.matmul(psi_all, Ham2, transpose_b=True)
  n_Heff_0 += 0.5 * dt * dt * n_Heff2_all[0]
  n_Heff_M += 0.5 * dt * dt * n_Heff2_all[-1]
  n_Heff += dt * dt * n_Heff2_all[1:-1]

  # Calculate sums
  phi_phi = tf.reduce_sum(tf.square(tf.abs(psi_all)))
  Heff_samples = tf.concat(
      [n_Heff_0[tf.newaxis], n_Heff, n_Heff_M[tf.newaxis]], axis=0)
  Heff_total = tf.reduce_sum(tf.math.conj(psi_all) * Heff_samples)
  return tf.math.real(Heff_total) / phi_phi