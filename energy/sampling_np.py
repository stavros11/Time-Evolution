import numpy as np


def calculate_psi(psi, configs, times):
  M, Nstates = psi.shape
  M += -1
  N = int(np.log2(Nstates))
  configs_dec = (configs < 0).dot(2**np.arange(0, N))

  # shape (3, Nsamples)
  return np.concatenate((psi[np.clip(times-1, 0, M), configs_dec][np.newaxis],
                         psi[times, configs_dec][np.newaxis],
                         psi[np.clip(times+1, 0, M), configs_dec][np.newaxis]),
                        axis=0)


def vmc_energy(full_psi, configs, times, dt, h=0.5):
  """Calculates Clock energy using full wavefunction and samples.

  Args:
    full_psi: Full wavefunction of shape (M + 1, 2**N)
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
  M, Nstates = full_psi.shape
  M += -1
  N = int(np.log2(Nstates))

  # shape (3, Nsamples)
  psi = calculate_psi(full_psi, configs, times)
  # Find boundary indices [ind0, ind(M)]
  boundary_ind = [np.where(times == i)[0] for i in [0, M]]

  # H^0 term
  Heff0_samples = 2 - (psi[0] + psi[2]) / psi[1]
  Heff0_samples[boundary_ind[0]] = 1 - psi[2][boundary_ind[0]] / psi[1][boundary_ind[0]]
  Heff0_samples[boundary_ind[1]] = 1 - psi[0][boundary_ind[1]] / psi[1][boundary_ind[1]]

  # shape (Nsamples,)
  classical_energy = (configs[:, 1:] * configs[:, :-1]).sum(axis=1) + configs[:, 0] * configs[:, -1]
  X = np.zeros_like(psi)
  XZZ, XX = np.zeros_like(psi[1]), np.zeros_like(psi[1])
  for i in range(N):
    flipper = np.ones_like(configs[0])
    flipper[i] = -1
    # shape (3, Nsamples)
    psi_flipped = calculate_psi(full_psi, flipper[np.newaxis] * configs, times)
    X += psi_flipped / psi[1][np.newaxis]
    XZZ += (classical_energy - 2 * configs[:, i] * (configs[:, (i-1)%N] +
            configs[:, (i+1)%N])) * psi_flipped[1] / psi[1]
    for j in range(N):
      flipper[j] *= -1
      psi_flipped2 = calculate_psi(full_psi, flipper[np.newaxis] * configs, times)
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


def vmc_gradients(psi, configs, times, dt, h=0.5, stoch_rec=False):
  ## Heff_samples has shape (Nsamples,)
  M, Nstates = psi.shape
  N = int(np.log2(Nstates))
  M += -1

  Heff_vmc, Heff_std, Heff_samples = vmc_energy(psi, configs, times, dt, h=0.5)

  # shape (Nsamples, Nstates)
  gradient_samples = np.zeros((len(configs), Nstates), dtype=psi.dtype)
  configs_dec = (configs < 0).dot(2**np.arange(0, N))
  gradient_samples[np.arange(len(configs)), configs_dec] = 1.0 / psi[times, configs_dec]
  gradient_star_Heff_samples = np.conj(gradient_samples) * Heff_samples[:, np.newaxis]

  # shape (M, Nstates) - Slow calculation
  Ok, Ok_star_Eloc = np.zeros_like(psi), np.zeros_like(psi)
  Ok_star_Ok = np.zeros(2*Ok.shape, dtype=Ok.dtype)
  for n in range(1, M+1):
    indices = np.where(times == n)
    Ok[n] = gradient_samples[indices].sum(axis=0)
    Ok_star_Eloc[n] = gradient_star_Heff_samples[indices].sum(axis=0)
    if stoch_rec:
      Ok_star_Ok[n, :, n, :] = (np.conj(gradient_samples)[indices][:, :, np.newaxis] *
                                gradient_samples[indices][:, np.newaxis]).sum(axis=0)

  return (Ok[1:] / len(configs), Ok_star_Eloc[1:] / len(configs), Heff_samples.mean(),
          Heff_vmc, Heff_std, Ok_star_Ok[1:, :, 1:])