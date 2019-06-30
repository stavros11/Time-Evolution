"""Calculates Clock energies and gradients. NumPy implementation.

Requires the full wavefunction at each time step and uses all states to
calculate expectation values deterministically.
(exponentially expensive sum - no sampling!)
"""

import numpy as np
import itertools


def all_states_Heff(psi_all, Ham, dt, phi_phi=None, Ham2=None, psi0=None):
  """Calculates Clock's expectation value.

  Args:
    psi_all: Full wavefunction at each time step of shape (M+1, 2**N)
      where N is the number of sites and M+1 the number of time steps.
    Ham: Full (real space) Hamiltonian matrix of shape (2**N, 2**N).
    dt: Time step.
    Ham2: Square of spin Hamiltonian of shape (2**N, 2**N).
    phi_phi: Norm of the full wave function.
    psi0: If given an initial condition penalty term is added to the
      Hamiltonian. This is typically not needed in variational approaches,
      since we impose the initial condition by fixing the initial parameters.

  If Ham2 or phi_phi are not given, they are calculated from Ham and psi_all.
  We allow the user to give them to speed up the calculation
  (avoid calculating them multiple times).

  Returns:
    Heff_exact: List with the three different Clock terms (groupped according
      to dt). The total <Clock> is the sum of these three terms.
    Heff_samples: Array of shape (M+1, 2**N) with the Hamiltonian terms
      before the sums.
  """
  M = len(psi_all) - 1

  if phi_phi is None:
    phi_phi = (np.abs(psi_all)**2).sum()

  if Ham2 is None:
    Ham2 = Ham.dot(Ham)

  # H^0 term
  n_Heff0 = np.zeros_like(psi_all)
  n_Heff0[0] = psi_all[0] - psi_all[1]
  n_Heff0[M] = psi_all[M] - psi_all[M-1]
  n_Heff0[1:M] = 2*psi_all[1:M] - psi_all[:M-1] - psi_all[2:]
  Heff_exact = [((np.conj(psi_all) * n_Heff0).sum() / phi_phi)]

  # H^1 term
  n_Heff1 = np.zeros_like(psi_all)
  n_Heff1[0] = - Ham.dot(psi_all[1])
  n_Heff1[M] = Ham.dot(psi_all[M-1])
  n_Heff1[1:M] = Ham.dot((psi_all[:M-1] - psi_all[2:]).T).T
  Heff_exact.append(1j * (np.conj(psi_all) * n_Heff1).sum() * dt / phi_phi)

  # H^2 term
  n_Heff2 = Ham2.dot(psi_all.T).T
  n_Heff2[0] *= 0.5
  n_Heff2[M] *= 0.5
  Heff_exact.append((np.conj(psi_all) * n_Heff2).sum() * dt * dt / phi_phi)

  Heff_samples = n_Heff0 + 1j * n_Heff1 * dt + n_Heff2 * dt * dt
  # Initial condition penalty term
  if psi0 is not None:
    Heff_samples[0] += psi_all[0] - (psi0.conj() * psi_all[0]).sum() * psi0

  return Heff_exact, Heff_samples


def all_states_gradient(full_psi, Ham, dt, Ham2=None, psi0=None):
  """Gradients of the Clock Hamiltonian with respect to a full wavefunction.

  Args:
    full_psi: Full wavefunction at each time step of shape (M + 1, 2**N)
      where N is the number of sites and M+1 the number of time steps.
    Ham: Full (real space) Hamiltonian matrix of shape (2**N, 2**N).
    dt: Time step.
    Ham2: Square of spin Hamiltonian of shape (2**N, 2**N).
      If Ham2 is not given then it is calculated from Ham.
    psi0: If given then the initial variational parameters are included in
      the gradient. This is typically not needed in variational approaches,
      since we impose the initial condition by fixing the initial parameters.

  Returns:
    Ok: EV of gradient of Clock with respect to every wavefunction component.
    Ok_star_Eloc: EV of Ok*Eloc term with respect to every wavefunc. component.
    Eloc: EV of Eloc (number).
    Eloc_terms: List of EVs of the three Clock terms.

  Ok and Ok_star_Eloc have shape (M, 2**N) (or (M + 1, 2**N) if psi0 is given).
  """
  M, Nstates = full_psi.shape
  M += -1

  if Ham2 is None:
    Ham2 = Ham.dot(Ham)

  phi_phi = (np.abs(full_psi)**2).sum()
  Eloc_terms, Heff_samples = all_states_Heff(full_psi, Ham, dt,
                                             phi_phi=phi_phi, Ham2=Ham2,
                                             psi0=psi0)

  Eloc = (np.conj(full_psi) * Heff_samples).sum() / phi_phi

  # shape (M, Nstates)
  Ok = np.conj(full_psi) / phi_phi
  Ok_star_Eloc = Heff_samples / phi_phi
  if psi0 is None:
    Ok = Ok[1:]
    Ok_star_Eloc = Ok_star_Eloc[1:]

  return Ok, Ok_star_Eloc, Eloc, Eloc_terms


def all_states_sampling_gradient(machine, Ham, dt, Ham2=None):
  """Use only to test the machines written for sampling purposes."""
  N, M = machine.n_sites, machine.time_steps
  if Ham2 is None:
    Ham2 = Ham.dot(Ham)

  full_psi = machine.dense()
  phi_phi = (np.abs(full_psi)**2).sum()
  Eloc_terms, Heff_samples = all_states_Heff(full_psi, Ham, dt,
                                             phi_phi=phi_phi, Ham2=Ham2)
  Eloc = (np.conj(full_psi) * Heff_samples).sum() / phi_phi

  all_configs = np.array(list(itertools.product([1, -1], repeat=N)))
  n_states = len(all_configs)

  times = np.repeat(np.arange(M + 1), n_states)
  configs = np.concatenate((M + 1) * [all_configs], axis=0)

  psi_samples = machine.wavefunction(configs, times)[1]
  grad_samples = machine.gradient(configs, times)

  slicer = (slice(None),) + (len(machine.shape) - 1) * (np.newaxis,)
  weights = np.abs(psi_samples[slicer])**2 * grad_samples

  weights = weights.reshape((M + 1, n_states) + machine.shape[1:])
  Ok = weights.sum(axis=1) / phi_phi
  slicer = (slice(None),) + slicer
  Heff_samples = Heff_samples / full_psi
  Ok_star_Eloc = (np.conj(weights) * Heff_samples[slicer]).sum(axis=1) / phi_phi

  return Ok[1:], Ok_star_Eloc[1:], Eloc, Eloc_terms
