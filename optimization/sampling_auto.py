"""Methods for VMC optimization of Clock using sampling.

TFIM model is assumed!
"""
import numpy as np
import tensorflow as tf
from machines import base
from typing import List, Tuple


def energy(machine: base.BaseMachine, configs: tf.Tensor, times: tf.Tensor,
           dt: float, h: float = 0.5
           ) -> Tuple[List[float], List[float], np.ndarray]:
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
  times_before = tf.clip_by_value(times - 1, 0, M)
  times_after = tf.clip_by_value(times + 1, 0, M)
  psi = np.exp(np.array([machine.forward_log(configs, times_before).numpy(),
                         machine.forward_log(configs, times).numpy(),
                         machine.forward_log(configs, times_after).numpy()]))

  # Find boundary indices [ind0, ind(M)]
  bind0, bindT = tuple(np.where(times.numpy() == i)[0] for i in [0, M])

  # H^0 term
  Heff0_samples = 2 - (psi[0] + psi[2]) / psi[1]

  Heff0_samples[bind0] = 1 - psi[2][bind0] / psi[1][bind0]
  Heff0_samples[bindT] = 1 - psi[0][bindT] / psi[1][bindT]

  # shape (Nsamples,)
  configs_np = configs.numpy()
  classical_energy = (configs_np[:, 1:] * configs_np[:, :-1]).sum(axis=1)
  classical_energy += configs_np[:, 0] * configs_np[:, -1]
  X = np.zeros_like(psi)
  XZZ, XX = np.zeros_like(psi[1]), np.zeros_like(psi[1])
  for i in range(N):
    flipper = np.ones_like(configs_np[0])
    flipper[i] = -1
    # shape (3, Nsamples)
    psi_flipped = np.exp(
        machine.forward_log(flipper[np.newaxis] * configs, times).numpy())
    X += psi_flipped / psi[1][np.newaxis]
    XZZ += (classical_energy - 2 * configs_np[:, i] * (configs_np[:, (i-1)%N] +
            configs_np[:, (i+1)%N])) * psi_flipped[1] / psi[1]
    for j in range(N):
      flipper[j] *= -1
      psi_flipped2 = np.exp(
          machine.forward_log(flipper[np.newaxis] * configs, times).numpy())
      flipper[j] *= -1
      XX += psi_flipped2[1] / psi[1]
  ZZX = classical_energy * X[1]

  # H^1 term
  # shape (3, Nsamples)
  Eloc = -classical_energy * psi / psi[1][np.newaxis] - h * X
  # shape (Nsamples,)
  Heff1_samples = Eloc[0] - Eloc[2]
  Heff1_samples[bind0] = -Eloc[2][bind0]
  Heff1_samples[bindT]= Eloc[0][bindT]
  Heff1_samples *= 1j * dt

  # H^2 term
  # shape (Nsamples,)
  Heff2_samples = dt * dt * (classical_energy**2 + h * ZZX + h * XZZ + h**2 * XX)
  Heff2_samples[bind0] *= 0.5
  Heff2_samples[bindT] *= 0.5

  Heff_vmc = [Heff0_samples.mean(), Heff1_samples.mean(), Heff2_samples.mean()]
  Heff_std = [[Heff0_samples.real.std(), Heff0_samples.imag.std()],
              [Heff1_samples.real.std(), Heff1_samples.imag.std()],
              [Heff2_samples.real.std(), Heff2_samples.imag.std()]]

  Heff_samples = Heff0_samples + Heff1_samples + Heff2_samples

  return Heff_vmc, Heff_std, Heff_samples


def gradient(machine: base.BaseMachine, configs: np.ndarray, times: np.ndarray,
             dt: float, h: float = 0.5
             ) -> Tuple[np.ndarray, np.ndarray, float,
                        List[float], List[float]]:
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
  with tf.GradientTape(persistent=True) as tape:
    tape.watch(machine.variables)
    logpsi = machine.forward_log(configs, times) # This has shape (Ns,)?
    heff_mean, heff_std, heff_samples = energy(machine, configs, times, dt, h)

  logpsi_av = tf.reduce_mean(logpsi)
  ok_re = tape.gradient(tf.real(logpsi_av), machine.variables)
  ok_im = tape.gradient(tf.imag(logpsi_av), machine.variables)
  ok = tf.complex(ok_re, ok_im)

  eloc_logpsi_av = tf.reduce_mean(tf.conj(logpsi) * heff_samples)
  ok_star_eloc_re = tape.gradient(tf.real(eloc_logpsi_av), machine.variables)
  ok_star_eloc_im = tape.gradient(tf.imag(eloc_logpsi_av), machine.variables)
  ok_star_eloc = tf.complex(ok_star_eloc_re, ok_star_eloc_im)

  return (ok, ok_star_eloc, heff_samples.mean(), heff_mean, heff_std)