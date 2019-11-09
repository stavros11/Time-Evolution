import numpy as np
from machines.autograd import full
from samplers import tfsamplers
from utils import tfim
import time

n_sites = 6
t_final = 1.0
time_steps = 20
h_init = 1.0
h_ev = 0.5


exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev)
init_state = exact_state[0]

machine = full.FullWavefunctionModel(init_state, time_steps, optimizer=None)


# Test machine's forward_log
psi = machine.forward_dense()
import itertools
all_confs = np.array(list(itertools.product([-1, 1], repeat=n_sites)))
logpsi = machine.forward_log(all_confs, np.zeros(len(all_confs)))

sampler = tfsamplers.SpinTime(machine, n_samples=5e3, n_corr=1, n_burn=10)

start_time = time.time()
samples = sampler()
print(samples.shape)
print(time.time() - start_time)