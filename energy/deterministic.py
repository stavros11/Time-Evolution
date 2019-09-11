"""Methods for VMC optimization of Clock using exact calculations.

Exact calculations = VMC with infinite samples.

Requires the full wavefunction at each time step and uses all states to
calculate expectation values deterministically.
(exponentially expensive sum - no sampling!)
Works with any model as it uses the full Hamiltonian matrix for the calculation.
"""
# Reminder: This script had a `normalized_clock` method implemented which was
# calculating a Clock derived from the projected Schrodinger's equation.
# This was removed because of change of the project's focus. If you need to
# recover it use the `master_0919patch` branch.
import numpy as np
import itertools
from machines import base
from typing import List, Optional, Tuple


def energy(psi_all: np.ndarray, ham: np.ndarray, dt: float,
           phi_phi: Optional[float] = None,
           ham2: Optional[np.ndarray] = None,
           psi0: Optional[np.ndarray] = None) -> Tuple[List[float], np.ndarray]:
  """Calculates Clock's expectation value.

  Args:
    psi_all: Full wavefunction at each time step of shape (M+1, 2**N)
      where N is the number of sites and M+1 the number of time steps.
    ham: Full (real space) Hamiltonian matrix of shape (2**N, 2**N).
    dt: Time step.
    ham2: Square of spin Hamiltonian of shape (2**N, 2**N).
    phi_phi: Norm of the full wave function.
    psi0: If an initial condition is given as `psi0`, a penalty term is added
      to the Hamiltonian. This is usually not needed in variational approaches,
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

  if ham2 is None:
    ham2 = ham.dot(ham)

  # H^0 term
  n_Heff0 = np.zeros_like(psi_all)
  n_Heff0[0] = psi_all[0] - psi_all[1]
  n_Heff0[M] = psi_all[M] - psi_all[M-1]
  n_Heff0[1:M] = 2*psi_all[1:M] - psi_all[:M-1] - psi_all[2:]
  Heff_exact = [((np.conj(psi_all) * n_Heff0).sum() / phi_phi)]

  # H^1 term
  n_Heff1 = np.zeros_like(psi_all)
  n_Heff1[0] = - ham.dot(psi_all[1])
  n_Heff1[M] = ham.dot(psi_all[M-1])
  n_Heff1[1:M] = ham.dot((psi_all[:M-1] - psi_all[2:]).T).T
  Heff_exact.append(1j * (np.conj(psi_all) * n_Heff1).sum() * dt / phi_phi)

  # H^2 term
  n_Heff2 = ham2.dot(psi_all.T).T
  n_Heff2[0] *= 0.5
  n_Heff2[M] *= 0.5
  Heff_exact.append((np.conj(psi_all) * n_Heff2).sum() * dt * dt / phi_phi)

  Heff_samples = n_Heff0 + 1j * n_Heff1 * dt + n_Heff2 * dt * dt
  # Initial condition penalty term
  if psi0 is not None:
    Heff_samples[0] += psi_all[0] - (psi0.conj() * psi_all[0]).sum() * psi0

  return Heff_exact, Heff_samples


def gradient(machine: base.BaseMachine,
             ham: np.ndarray, dt: float,
             ham2: Optional[np.ndarray] = None,
             psi0: Optional[np.ndarray] = None
             ) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
  """Gradients of the Clock Hamiltonian with respect to a full wavefunction.

  Args:
    machine: Machine to get `full_psi` from. Note that only the dense
      wavefunction is used here, however we use machine as an argument
      to be consistent with `sampling_gradient` method.
    ham: Full (real space) Hamiltonian matrix of shape (2**N, 2**N).
    dt: Time step.
    ham2: Square of spin Hamiltonian of shape (2**N, 2**N).
      If Ham2 is not given then it is calculated from Ham.
    psi0: If an initial condition is given as `psi0`, a penalty term is added
      to the Hamiltonian. This is usually not needed in variational approaches,
      since we impose the initial condition by fixing the initial parameters.

  Returns:
    Ok: EV of gradient of Clock with respect to every wavefunction component.
    Ok_star_Eloc: EV of Ok*Eloc term with respect to every wavefunc. component.
    Eloc: EV of Eloc (number).
    Eloc_terms: List of EVs of the three Clock terms.

  Ok and Ok_star_Eloc have shape (M, 2**N) (or (M + 1, 2**N) if psi0 is given).
  """
  full_psi = machine.dense()
  M, Nstates = full_psi.shape
  M += -1

  if ham2 is None:
    ham2 = ham.dot(ham)

  phi_phi = (np.abs(full_psi)**2).sum()
  Eloc_terms, Heff_samples = energy(full_psi, ham, dt, phi_phi=phi_phi,
                                    ham2=ham2, psi0=psi0)

  Eloc = (np.conj(full_psi) * Heff_samples).sum() / phi_phi

  # shape (M, Nstates)
  Ok = np.conj(full_psi) / phi_phi
  Ok_star_Eloc = Heff_samples / phi_phi
  if psi0 is None:
    Ok = Ok[1:]
    Ok_star_Eloc = Ok_star_Eloc[1:]

  return Ok, Ok_star_Eloc, Eloc, Eloc_terms


def sampling_gradient(machine: base.BaseMachine,
                      ham: np.ndarray, dt: float,
                      ham2: Optional[np.ndarray] = None
                      ) -> Tuple[np.ndarray, np.ndarray, float, List[float]]:
  """Determinisitc calculation but using the 'sampling way'.

  Useful to test machines that are originally designed for sampling.
  """
  N, M = machine.n_sites, machine.time_steps
  if ham2 is None: ham2 = ham.dot(ham)

  full_psi = machine.dense()
  phi_phi = (np.abs(full_psi)**2).sum()

  Eloc_terms, Heff_samples = energy(full_psi, ham, dt, phi_phi=phi_phi,
                                    ham2=ham2)
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