import numpy as np


def vmc_energy(machine, configs, times, dt, h=0.5):
  """Calculates Clock energy using samples.

  Args:
    machine: Machine object (see /machines/base.py)
    configs: Spin configuration samples of shape (Ns, N)
    times: Time configuration samples of shape (Ns,)
    dt: Time step
    h: Field of evolution TFIM Hamiltonian.
  here N = number of sites, M=time steps, Ns=number of samples.

  Returns:
    Heff_vmc: List with the average three terms of the Clock Hamiltonian.
    Heff_std: List with the STD of the three terms of the Clock Hamiltonian.
    Heff_samples: Samples of the Clock Hamiltonian of shape (Ns,)
  """
  M, N = machine.time_steps, machine.n_sites

  # shape (3, Nsamples)
  psi = machine.wavefunction(configs, times)
  # Find boundary indices [ind0, ind(M)]
  boundary_ind = [np.where(times == i)[0] for i in [0, M]]

  # H^0 term
  Heff0_samples = 2 - (psi[0] + psi[2]) / psi[1]
  Heff0_samples[boundary_ind[0]] = 1 - psi[2][boundary_ind[0]] / psi[1][boundary_ind[0]]
  Heff0_samples[boundary_ind[1]] = 1 - psi[0][boundary_ind[1]] / psi[1][boundary_ind[1]]

  # shape (Nsamples,)
  classical_energy = (configs[:, 1:] * configs[:, :-1]).sum(axis=1)
  classical_energy += configs[:, 0] * configs[:, -1]
  X = np.zeros_like(psi)
  XZZ, XX = np.zeros_like(psi[1]), np.zeros_like(psi[1])
  for i in range(N):
    flipper = np.ones_like(configs[0])
    flipper[i] = -1
    # shape (3, Nsamples)
    psi_flipped = machine.wavefunction(flipper[np.newaxis] * configs, times)
    X += psi_flipped / psi[1][np.newaxis]
    XZZ += (classical_energy - 2 * configs[:, i] * (configs[:, (i-1)%N] +
            configs[:, (i+1)%N])) * psi_flipped[1] / psi[1]
    for j in range(N):
      flipper[j] *= -1
      psi_flipped2 = machine.wavefunction(flipper[np.newaxis] * configs, times)
      flipper[j] *= -1
      XX += psi_flipped2[1] / psi[1]
  ZZX = classical_energy * X[1]

  # H^1 term
  # shape (3, Nsamples)
  Eloc = -classical_energy * psi / psi[1][np.newaxis] - h * X
  # shape (Nsamples,)
  Heff1_samples = Eloc[0] - Eloc[2]
  Heff1_samples[boundary_ind[0]] = -Eloc[2][boundary_ind[0]]
  Heff1_samples[boundary_ind[1]]= Eloc[0][boundary_ind[1]]
  Heff1_samples *= 1j * dt

  # H^2 term
  # shape (Nsamples,)
  Heff2_samples = dt * dt * (classical_energy**2 + h * ZZX + h * XZZ + h**2 * XX)
  Heff2_samples[boundary_ind[0]] *= 0.5
  Heff2_samples[boundary_ind[1]] *= 0.5

  Heff_vmc = [Heff0_samples.mean(), Heff1_samples.mean(), Heff2_samples.mean()]
  Heff_std = [[Heff0_samples.real.std(), Heff0_samples.imag.std()],
              [Heff1_samples.real.std(), Heff1_samples.imag.std()],
              [Heff2_samples.real.std(), Heff2_samples.imag.std()]]

  Heff_samples = Heff0_samples + Heff1_samples + Heff2_samples

  return Heff_vmc, Heff_std, Heff_samples


def vmc_gradients(machine, configs, times, dt, h=0.5):
  """Calculates gradients using samples.

  Assumes that the machine has the same form at each time step, that is we
  use the same ansatz in every time step - we do not have explicit time
  dependence in the ansatz or the parameters.

  Args:
    machine: Machine object (see /machines/base.py)
    configs: Spin configuration samples of shape (Ns, N)
    times: Time configuration samples of shape (Ns,)
    dt: Time step
    h: Field of evolution TFIM Hamiltonian.
  here N = number of sites, M=time steps, Ns=number of samples.

  Returns:
    Ok: <Ok> of the shape of variational parameters
    Ok_star_Eloc: <Ok*Eloc> of the shape of variational parameters
    Eloc: <Eloc> as number
    Heff_vmc: List with the average three terms of the Clock Hamiltonian.
    Heff_std: List with the STDs of the three terms of the Clock Hamiltonian.
  """
  ## Heff_samples has shape (Nsamples,)
  Heff_vmc, Heff_std, Heff_samples = vmc_energy(machine, configs, times, dt, h=h)

  # shape (Nsamples, Nstates)
  grad_samples = machine.gradient(configs, times)
  slicer = (slice(None),) + (len(grad_samples.shape) - 1) * (np.newaxis,)
  gradient_star_Heff_samples = np.conj(grad_samples) * Heff_samples[slicer]

  # shape (M, Nstates) - Slow calculation
  dtype = grad_samples.dtype
  shape = (machine.time_steps,) + grad_samples.shape[1:]
  Ok, Ok_star_Eloc = np.zeros(shape, dtype=dtype), np.zeros(shape, dtype=dtype)
  for n in range(machine.time_steps):
    indices = np.where(times == n + 1)
    Ok[n] = grad_samples[indices].sum(axis=0)
    Ok_star_Eloc[n] = gradient_star_Heff_samples[indices].sum(axis=0)

  return (Ok / len(configs), Ok_star_Eloc / len(configs),
          Heff_samples.mean(), Heff_vmc, Heff_std)