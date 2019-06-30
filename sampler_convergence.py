"""Tests convergence of the C++ samplers found in /sampler directory."""

import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import utils
from numpy.ctypeslib import ndpointer
from energy import full_np
from energy import sampling_np
from machines import full

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 22})



# System parameters
n_sites = 4
time_steps = 40
t_final = 1.0
h_init = 1.0
h_ev = 0.5
n_samples_list = [1000, 1500, 3000, 5000, 10000, 15000, 20000,
                  25000, 30000, 35000, 40000, 45000, 50000, 60000]


# Find exact evolution state
t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)
ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)

# State to use for sampling
machine = full.FullWavefunctionMachine(exact_state[0], time_steps)
machine.set_parameters(exact_state.reshape(
    (time_steps + 1,) + machine.shape[1:]))

# Calculate energy exactly
exact_energy, _ = full_np.all_states_Heff(exact_state, ham, dt, Ham2=ham2)


# Load sampler
sampler = ctypes.CDLL(os.path.join(os.getcwd(), "samplers", "qtvmclib.so"))
sampler.run.argtypes = ([ndpointer(np.complex128, flags="C_CONTIGUOUS")] + 6 * [ctypes.c_int] +
                        [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                         ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")])
sampler.run.restype = None

energy_means = [[], [], []]
energy_stds = [[], [], []]
for n_samples in n_samples_list:
  # Sample
  configs = np.zeros([n_samples, n_sites], dtype=np.int32)
  times = np.zeros(n_samples, dtype=np.int32)
  sampler.run(machine.dense(), n_sites, time_steps + 1, 2**n_sites,
              n_samples, 1, 50, configs, times)

  # Calculate energy with sampling
  #samp_results = sampling_np.vmc_energy(machine, configs, times, dt, h=h_ev)
  samp_results = sampling_np.vmc_gradients(machine, configs, times, dt, h=h_ev)
  for i in range(3):
    energy_means[i].append(samp_results[-2][i])
    energy_stds[i].append(samp_results[-1][i])

exact_energy = np.array(exact_energy)
energy_means = np.array(energy_means)
energy_stds = np.array(energy_stds)

# Plots
label_size = 36
plt.figure(figsize=(32, 16))
for i in range(3):
  plt.subplot(331 + 3*i)
  plt.plot(n_samples_list, energy_means[i].real, '-o', color='red', linewidth=2.5, markersize=12)
  plt.axhline(y=exact_energy[i].real, linestyle='--', color='blue', linewidth=2.5)
  #plt.yticks([])
  if i == 2:
    plt.xlabel(r'$N_\mathrm{samples}$', fontsize=label_size)
  plt.ylabel('Real', fontsize=label_size)

  plt.subplot(332 + 3*i)
  plt.plot(n_samples_list, energy_means[i].imag, '-o', color='red', linewidth=2.5, markersize=12)
  plt.axhline(y = exact_energy[i].imag, linestyle='--', color='blue', linewidth=2.5)
  #plt.yticks([])
  if i == 2:
    plt.xlabel(r'$N_\mathrm{samples}$', fontsize=label_size)
  plt.ylabel('Imag', fontsize=label_size)

  plt.subplot(333 + 3*i)
  plt.plot(n_samples_list, energy_stds[i, :, 0], '-o', color='red', linewidth=2.5, markersize=12)
  #plt.yticks([])
  if i == 2:
    plt.xlabel(r'$N_\mathrm{samples}$', fontsize=label_size)
  plt.ylabel('Re STD', fontsize=label_size)

plt.show()
#plt.savefig('convergence_test.pdf', bbox_inches='tight')
