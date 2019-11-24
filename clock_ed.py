import numpy as np
from utils import tfim
from utils import ed
from utils import calc


n_sites = 6
time_steps = 20
t_final = 1.0
h_init, h_ev = 1.0, 0.5


t_grid = np.linspace(0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]
ham = tfim.tfim_hamiltonian(n_sites, h=h_ev, pbc=True)
exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev)


clock = ed.construct_sparse_clock(ham, dt, time_steps, init_penalty=0.01,
                                  psi0=np.copy(exact_state[0]))
print(clock.shape)
print("\n\n")

eigvals, eigvecs = np.linalg.eigh(np.array(clock.todense()))
print(eigvals.shape)
print(eigvecs.shape)
print("\n\n")


print(eigvals[:10])
ground_state = eigvecs[:, 0].reshape(exact_state.shape)

norm = (np.abs(ground_state)**2).sum(axis=1)
print("\nNorm:", norm.mean(), norm.std())
print(norm)

overlap_t = calc.time_overlap(ground_state, exact_state)
print("\nOverlaps:")
print(overlap_t)