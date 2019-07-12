"""Tests for machines/mps_trotter.py"""

import numpy as np
import utils
from machines import mps_utils
from machines import mps_trotter


n_sites = 6
h_init = 1.0
h_ev = 0.5
time_steps = 100
t = np.linspace(0.0, 1.0, time_steps + 1)
dt = t[1] - t[0]
d_bond = 6

exact_state, exact_obs = utils.tfim_exact_evolution(n_sites, t[-1], time_steps, h0=h_init, h=h_ev)

mps = mps_trotter.TFIMTrotterMPS(exact_state[0], d_bond, dt, h=h_ev)
mps_state = mps.dense_evolution(time_steps)

ux = mps_utils.mpo_to_dense(mps.x_op[:, np.newaxis, :, :, np.newaxis])
uz = mps_utils.mpo_to_dense(mps.zz_op)
ident = np.eye(ux.shape[0], dtype=ux.dtype)

diff1 = ux.dot(ux.conj().T) - ident
diff2 = ux.conj().T.dot(ux) - ident
print(diff1.mean(), diff1.real.std(), diff1.imag.std())
print(diff2.mean(), diff2.real.std(), diff2.imag.std())

diff1 = uz.dot(uz.conj().T) - ident
diff2 = uz.conj().T.dot(uz) - ident
print(diff1.mean(), diff1.real.std(), diff1.imag.std())
print(diff2.mean(), diff2.real.std(), diff2.imag.std())