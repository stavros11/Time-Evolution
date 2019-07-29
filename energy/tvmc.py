"""Methods for applying traditional t-VMC updates."""

import itertools
import numpy as np
from scipy.sparse import linalg


def vmc_energy(machine, configs, h, psi=None):
  if psi is None:
    psi = machine.wavefunction(configs)

  # shape (Nsamples,)
  classical_energy = (configs[:, 1:] * configs[:, :-1]).sum(axis=1)
  classical_energy += configs[:, 0] * configs[:, -1]

  X = np.zeros_like(psi)
  for i in range(machine.n_sites):
    flipper = np.ones_like(configs[0])
    flipper[i] = -1
    # shape (3, Nsamples)
    psi_flipped = machine.wavefunction(flipper[np.newaxis] * configs)
    X += psi_flipped / psi

  # shape (Nsamples,)
  return -classical_energy - h * X


def exact_tvmc_step(machine, h=0.5):
  all_configs = list(itertools.product([-1, 1], repeat=machine.n_sites))
  all_configs = np.array(all_configs)

  psi = machine.dense()
  psi2 = np.abs(psi)**2
  norm2 = psi2.sum()

  Eloc_samples = vmc_energy(machine, all_configs, h=h)
  Eloc = (psi2 * Eloc_samples).mean() / norm2

  # grad samples with shape (Nsamples,) + machine.shape
  grads = machine.gradient(all_configs)
  flattened_shape = (len(grads), np.prod(grads.shape[1:]))
  grads = grads.reshape(flattened_shape)
  Ok = (psi2 * grads).mean(axis=0) / norm2

  Ok_star_Eloc = (psi2[:, np.newaxis] * Eloc_samples[:, np.newaxis]
                  * grads.conj()).mean(axis=0) / norm2
  Fk = Ok_star_Eloc - Ok.conj() * Eloc

  Ok_star_Ok = (psi2[:, np.newaxis, np.newaxis] *
                grads[:, :, np.newaxis].conj() *
                grads[:, np.newaxis]).mean(axis=0)
  Skk = Ok_star_Ok - np.outer(Ok.conj(), Ok)

  rhs = linalg.gmres(Skk, Fk)
  return rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok


def sampling_tvmc_step(machine, configs, h=0.5):
  """Calculates quantities needed to evolve for one t-VMC step.

  Args:
    machine: A 'step' machine object that inherits `BaseStepMachine`.
    configs: Array with sampled spin configs with shape (Ns, N).
    h: Value of the magnetic field for energy calculation.

  Returns:
    rhs: RHS of the evolution equation with shape VarPar.
    Ok: EV of gradients <Ok> with shape VarPar.
    Ok_star_Eloc: <OkEloc> with shape VarPar.
    Eloc: Local energy <H> at the current time (float).
    Ok_star_Eloc: <Ok*Ok> with shape VarPar + VarPar.

  Here VarPar = machine.shape
  """
  # TODO: Remove `exact_tvmc_step` and add its functionality in
  # the current method when configs=None is given.
  Eloc_samples = vmc_energy(machine, configs, h=h)
  Eloc = Eloc_samples.mean()

  # grad samples with shape (Nsamples,) + machine.shape
  grads = machine.gradient(configs)
  flattened_shape = (len(grads), np.prod(grads.shape[1:]))
  grads = grads.reshape(flattened_shape)
  Ok = grads.mean(axis=0)

  Ok_star_Eloc = (Eloc_samples[:, np.newaxis] * grads.conj()).mean(axis=0)
  Fk = Ok_star_Eloc - Ok.conj() * Eloc

  Ok_star_Ok = (grads[:, :, np.newaxis].conj() *
                grads[:, np.newaxis]).mean(axis=0)
  Skk = Ok_star_Ok - np.outer(Ok.conj(), Ok)

  rhs = linalg.gmres(Skk, Fk)
  return rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok