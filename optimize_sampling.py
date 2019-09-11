"""Optimizes full wavefunction using numpy.

Uses sampling to calculate gradients.
"""

import os
import ctypes
import h5py
import numpy as np
import matplotlib.pyplot as plt
import utils
from energy import full_np
from energy import sampling_np
from machines import full, mps
from numpy.ctypeslib import ndpointer


# Model parameters
n_sites = 6
time_steps = 20
t_final = 1.0
h_init = 1.0
h_ev = 0.5
sample_time = True

# Optimization parameters
n_epochs = 10000
n_message = 200

# Sampling parameters (per time when using the space only sampler)
n_samples = 20000
n_corr = 1
n_burn = 10

t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]

ham = utils.tfim_hamiltonian(n_sites, h=h_ev)
ham2 = ham.dot(ham)
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)

# Initialize machine
#machine = full.FullWavefunctionMachine(exact_state[0], time_steps)
machine = mps.SmallMPSMachine(exact_state[0], time_steps, d_bond=5)
optimizer = utils.AdamComplex(machine.shape, dtype=machine.dtype)

# Initialize sampler
if sample_time:
  sampler = ctypes.CDLL(os.path.join(os.getcwd(), "samplers", "qtvmclib.so"))
  sampler.run.argtypes = ([ndpointer(np.complex128, flags="C_CONTIGUOUS")] +
                           6 * [ctypes.c_int] +
                          [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                           ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")])
  sampler.run.restype = None
  configs = np.zeros([n_samples, n_sites], dtype=np.int32)
  times = np.zeros(n_samples, dtype=np.int32)

else:
  sampler = ctypes.CDLL(os.path.join(os.getcwd(), "samplers", "spacevmclib.so"))
  sampler.run.argtypes = ([ndpointer(np.complex128, flags="C_CONTIGUOUS")] +
                           6 * [ctypes.c_int] +
                          [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")])
  sampler.run.restype = None
  configs = np.zeros([n_samples * (time_steps + 1), n_sites], dtype=np.int32)
  times = np.repeat(np.arange(time_steps + 1), n_samples).astype(np.int32)


history = {"overlaps": [], "avg_overlaps": [],
           "exact_Eloc": [], "sampled_Eloc": []}
for epoch in range(n_epochs):
  # Sample
  if sample_time:
    sampler.run(machine.dense(), n_sites, time_steps + 1, 2**n_sites,
                n_samples, n_corr, n_burn, configs, times)
  else:
    sampler.run(machine.dense(), n_sites, time_steps + 1, 2**n_sites,
                n_samples, n_corr, n_burn, configs)

  # Calculate gradient
  Ok, Ok_star_Eloc, Eloc, _, _ = sampling_np.vmc_gradients(machine, configs, times, dt, h=h_ev)
  grad = Ok_star_Eloc - Ok.conj() * Eloc
  # Update machine
  machine.update(optimizer.update(grad, epoch))

  # Calculate energy exactly (using all states) on the current machine state
  exact_Eloc, _ = full_np.all_states_Heff(machine.dense(), ham, dt, Ham2=ham2)
  exact_Eloc = np.array(exact_Eloc).sum()

  history["overlaps"].append(utils.overlap(machine.dense(), exact_state))
  history["avg_overlaps"].append(utils.averaged_overlap(machine.dense(), exact_state))
  history["exact_Eloc"].append(exact_Eloc)
  history["sampled_Eloc"].append(Eloc)
  if epoch % n_message == 0:
    Eloc_error = np.abs((Eloc - exact_Eloc) * 100.0 / exact_Eloc)
    print("\nEpoch: {}".format(epoch))
    print("Sampled Eloc: {}".format(Eloc))
    print("Exact Eloc: {}".format(exact_Eloc))
    print("Sampling/Exact Eloc error: {}%".format(Eloc_error))
    print("Overlap: {}".format(history["overlaps"][-1]))
    print("Averaged Overlap: {}".format(history["avg_overlaps"][-1]))

# Save history
sampling_type = ["space", ""][sample_time]
filename = "{}sampling{}_{}_N{}M{}.h5py".format(
    sampling_type, n_samples, machine.name, n_sites, time_steps)
file = h5py.File("histories/{}".format(filename), "w")
for k in history.keys():
  file[k] = history[k]
file.close()

# Save final dense wavefunction
filename = "{}.npy".format(filename[:-5])
np.save("final_dense/{}".format(filename), machine.dense())