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
  Ok = (psi2[:, np.newaxis] * grads).mean(axis=0) / norm2

  Ok_star_Eloc = (psi2[:, np.newaxis] * Eloc_samples[:, np.newaxis]
                  * grads.conj()).mean(axis=0) / norm2
  Fk = Ok_star_Eloc - Ok.conj() * Eloc

  Ok_star_Ok = (psi2[:, np.newaxis, np.newaxis] *
                grads[:, :, np.newaxis].conj() *
                grads[:, np.newaxis]).mean(axis=0)
  Skk = Ok_star_Ok - np.outer(Ok.conj(), Ok)

  rhs, info = linalg.gmres(Skk, Fk)
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
  def batched_mean(grads, size=100):
    assert len(grads) % size == 0
    n = len(grads) // size
    d = grads.shape[-1]
    s = np.zeros((d, d), dtype=grads.dtype)
    for i in range(n):
      s += (grads[i * size: (i + 1) * size, :, np.newaxis].conj() *
            grads[i * size: (i + 1) * size, np.newaxis]).sum(axis=0)
    return s / len(grads)

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

  Ok_star_Ok = batched_mean(grads)
  Skk = Ok_star_Ok - np.outer(Ok.conj(), Ok)

  rhs, info = linalg.gmres(Skk, Fk)
  return rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok


def evolve(machine, time_steps, dt, h=0.5, sampler=None):
  """Evolves an initialized machine.

  Args:
    machine: A machine object that inherits `BaseStepMachine`.
    time_steps: Number of steps to evolve (initial step not included).
    dt: Size of time steps.
    h: Value of magnetic field in evolution Hamiltonian.

  Returns:
    full_psi: The full wavefunction of the evolution with shape (T+1,2^N).
  """
  n_sites = machine.n_sites
  full_psi = [machine.dense()]
  if sampler is not None:
    configs = np.zeros([sampler.n_samples, n_sites], dtype=np.int32)

  for step in range(time_steps):
    if sampler is None:
      rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok = exact_tvmc_step(machine, h=h)
    else:
      sampler.run(machine.dense(), n_sites, 1, 2**n_sites, sampler.n_samples,
                  sampler.n_corr, sampler.n_burn, configs)
      rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok = sampling_tvmc_step(machine,
                                                                   configs,
                                                                   h=h)

    machine.update(-1j * rhs * dt)
    full_psi.append(machine.dense())
  return np.array(full_psi)