import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from machines.autograd import full
from samplers import tfsamplers
from optimization import deterministic_auto, sampling_auto
from utils import tfim
import time

n_sites = 4
t_final = 1.0
time_steps = 20
h_init = 1.0
h_ev = 0.5
dt = t_final / time_steps


exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev)
ham = tfim.tfim_hamiltonian(n_sites, h=h_ev, pbc=True)
init_state = exact_state[0]

machine = full.FullWavefunctionModel(init_state, time_steps, optimizer=None)
ham_tf = tf.convert_to_tensor(ham, dtype=machine.ctype)
machine.forward_dense()

# Exact calculation
exact_eloc = deterministic_auto.energy(machine.dense, ham_tf, dt).numpy()


# Sampling calculation
n_samples_list = [20, 200, 500, 1000, 2500, 5000, 10000]
sampling_results = []

for n_samples in n_samples_list:
  start_time = time.time()
  sampler = tfsamplers.SpinTime(machine, n_samples=n_samples, n_corr=1, n_burn=10)
  sampled_confs = sampler()
  configs, times = sampled_confs[:, :-1], sampled_confs[:, -1]
  sampling_results.append(sampling_auto.energy(machine, configs, times, dt, h_ev))
  print(configs.shape, times.shape, time.time() - start_time)

sampled_eloc = np.array([x[-1].mean() for x in sampling_results])

plt.figure(figsize=(7, 4))
plt.plot(n_samples_list, sampled_eloc, color="red", linewidth=2.0, marker="o", markersize=8, linestyle="--")
plt.axhline(y=exact_eloc, color="blue", linestyle="--", linewidth=2.0)
plt.xlabel("Number of samples")
plt.ylabel("$E_{loc}$")
plt.show()