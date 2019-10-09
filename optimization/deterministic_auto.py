"""Methods for VMC optimization of Clock using exact calculations.

Exact calculations = VMC with infinite samples.
Code is written in TensorFlow so that autograd can be used.

Requires the full wavefunction at each time step and uses all states to
calculate expectation values deterministically.
(exponentially expensive sum - no sampling!)
"""
import itertools
import numpy as np
import tensorflow as tf
from machines import autograd
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
  M = len(psi_all) - 1

  if Ham2 is None:
    Ham2 = tf.matmul(Ham, Ham)

  # H^0 term
  n_Heff_0 = psi_all[0] - psi_all[1]
  n_Heff_M = psi_all[M] - psi_all[M-1]
  n_Heff = 2 * psi_all[1:M] - psi_all[:M-1] - psi_all[2:]

  # H^1 term
  n_Heff_0 -= 1j * dt * tf.matmul(Ham, psi_all[1][:, tf.newaxis])[:, 0]
  n_Heff_M += 1j * dt * tf.matmul(Ham, psi_all[M-1][:, tf.newaxis])[:, 0]
  n_Heff += 1j * dt * tf.matmul(psi_all[:M - 1] - psi_all[2:], Ham,
                                transpose_b=True)

  # H^2 term
  n_Heff2_all = tf.matmul(psi_all, Ham2, transpose_b=True)
  n_Heff_0 += 0.5 * dt * dt * n_Heff2_all[0]
  n_Heff_M += 0.5 * dt * dt * n_Heff2_all[M]
  n_Heff += dt * dt * n_Heff2_all[1:M]

  # Calculate sums
  phi_phi = tf.reduce_sum(tf.square(tf.abs(psi_all)))
  Heff_samples = tf.concat(
      [n_Heff_0[tf.newaxis], n_Heff, n_Heff_M[tf.newaxis]], axis=0)
  Heff_total = tf.reduce_sum(tf.conj(psi_all) * Heff_samples)
  return tf.real(Heff_total) / phi_phi


def gradient(machine: autograd.BaseAutoGrad,
             ham: tf.Tensor, dt: float,
             ham2: Optional[tf.Tensor] = None
             ) -> Tuple[List[tf.Tensor], None, np.ndarray, None]:
  n_sites, time_steps = machine.n_sites, machine.time_steps
  all_configs = np.array(list(itertools.product([-1, 1], repeat=n_sites)))
  n_states = len(all_configs)

  times = np.repeat(np.arange(time_steps + 1), n_states)
  configs = np.concatenate((time_steps + 1) * [all_configs], axis=0)

  with tf.GradientTape() as tape:
    tape.watch(machine.variables)
    full_psi = tf.reshape(machine.wavefunction(configs, times),
                          machine.dense_shape)
    heff_ev = energy(full_psi, ham, dt, ham2)

  grad = tape.gradient(heff_ev, machine.variables)
  return grad, None, heff_ev.numpy(), None


