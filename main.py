import functools
import numpy as np
import tensorflow as tf
import clock
import machine
from utils import calc, tfim

n_sites = 6
time_steps = 20
t_final = 1.0
h_init = 1.0
h_ev = 0.5
n_epochs = 10000
n_message = 500

dt = t_final / time_steps
t = np.linspace(0, t_final, time_steps + 1)
assert dt == t[1] - t[0]

# Define system
ham = tfim.tfim_hamiltonian(n_sites, h=h_ev)
exact_state, exact_obs = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                                   h0=h_init, h=h_ev)

# Define ansatz
initial_state = np.array((time_steps + 1) * [exact_state[0]])
model_re = machine.FullWavefunction(initial_state.real)
model_im = machine.FullWavefunction(initial_state.imag)
model = machine.CartesianMachine(model_re, model_im)

ham_tf = tf.convert_to_tensor(ham, dtype=model.ctype)
ham2_tf = tf.matmul(ham, ham)
objective_func = functools.partial(clock.energy, Ham=ham_tf, dt=dt, Ham2=ham2_tf)


# Optimization
history = {"exact_Eloc": [], "avg_overlaps": []}
for epoch in range(n_epochs):
  energy = model.update(objective_func)
  history["exact_Eloc"].append(energy.numpy())
  history["avg_overlaps"].append(
        calc.averaged_overlap(model.dense.numpy(), exact_state))
  if epoch % n_message == 0:
    print("\nEpoch: {}".format(epoch))
    for k, v in history.items():
      print("{}: {}".format(k, v[-1]))
