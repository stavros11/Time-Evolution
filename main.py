import numpy as np
from utils import tfim


n_sites = 6
time_steps = 20
t_final = 1.0
h_init = 1.0
h_ev = 0.5

dt = t_final / time_steps
t = np.linspace(0, t_final, time_steps + 1)
assert dt == t[1] - t[0]

ham = tfim.tfim_hamiltonian(n_sites, h=h_ev)
exact_state = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                        h0=h_init, h=h_ev)

print(exact_state.shape)
