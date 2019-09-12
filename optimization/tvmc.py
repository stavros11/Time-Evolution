"""Methods for applying traditional t-VMC updates."""

import operator
import functools
import itertools
import numpy as np
from scipy.sparse import linalg


def vmc_energy(machine, configs, h):
  """Calculates samples of TFIM energy for given spin configurations.

  Args:
    machine: Machine object that inherits `BaseStepMachine`.
    configs: Spin configurations of shape (Ns, N).
    h: Value of magnetic field in evolution Hamiltonian.

  Returns:
    Energy samples of shape (Ns,).
  """
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


def batched_mean(grads, size=100):
  """Calculates <Ok*Ok> with batches to avoid MemoryError.

  Args:
    grads: The gradient shamples with shape (Ns, VarPar).
    size: Size of batches to use. Must be a divisor of len(grads).

  Returns:
    <Ok*Ok> of shape (VarPar, VarPar).
  """
  assert len(grads) % size == 0
  n = len(grads) // size
  d = grads.shape[-1]
  s = np.zeros((d, d), dtype=grads.dtype)
  for i in range(n):
    s += (grads[i * size: (i + 1) * size, :, np.newaxis].conj() *
          grads[i * size: (i + 1) * size, np.newaxis]).sum(axis=0)
  return s / len(grads)


def product_mean(*tensors):
  """Calculates mean of product of tensors.

  Mean is calculated over the axis=0."""
  return functools.reduce(operator.mul, tensors).mean(axis=0)


def sampling_tvmc_step(machine, configs=None, h=0.5):
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
  exact = configs is None
  if exact:
    configs = list(itertools.product([-1, 1], repeat=machine.n_sites))
    configs = np.array(configs)

    psi = machine.dense()
    psi2 = np.abs(psi)**2
    norm2 = psi2.sum()

    def prod_mean(*tensors):
      slicer = (slice(None),) + (len(tensors[0].shape) - 1) * (np.newaxis,)
      return product_mean(*tensors, psi2[slicer]) / norm2

    batch_mean = lambda grads: prod_mean(grads[:, :, np.newaxis].conj(),
                                         grads[:, np.newaxis])

  else:
    prod_mean = product_mean
    batch_mean = batched_mean


  Eloc_samples = vmc_energy(machine, configs, h=h)
  Eloc = prod_mean(Eloc_samples)

  # grad samples with shape (Nsamples,) + machine.shape
  grads = machine.gradient(configs)
  flattened_shape = (len(grads), np.prod(grads.shape[1:]))
  grads = grads.reshape(flattened_shape)
  Ok = prod_mean(grads)

  Ok_star_Eloc = prod_mean(Eloc_samples[:, np.newaxis], grads.conj())
  Ok_star_Ok = batch_mean(grads)

  Fk = Ok_star_Eloc - Ok.conj() * Eloc
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
  if sampler is None:
    step_calc = lambda m: sampling_tvmc_step(m, configs=None, h=h)
  else:
    configs = np.zeros([sampler.n_samples, n_sites], dtype=np.int32)
    step_calc = lambda m, c: sampling_tvmc_step(m, configs=c, h=h)

  for step in range(time_steps):
    if sampler is None:
      rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok = step_calc(machine)
    else:
      sampler.run(machine.dense(), n_sites, 1, 2**n_sites, sampler.n_samples,
                  sampler.n_corr, sampler.n_burn, configs)
      rhs, Ok, Ok_star_Eloc, Eloc, Ok_star_Ok = step_calc(machine, configs)

    machine.update(-1j * rhs * dt)
    full_psi.append(machine.dense())
  return np.array(full_psi)