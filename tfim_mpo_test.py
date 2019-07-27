"""Tests `tfim_mpo` method from `utils.py`."""

import utils
from machines import mps_utils

n_sites = 8
h = 1.0

ham = utils.tfim_hamiltonian(n_sites, h)
mpo = utils.tfim_mpo(n_sites, h)
print(mpo.shape)

dense = mps_utils.mpo_to_dense(mpo, trace=False)
print(dense.shape)

print()
diff = dense - ham
print(diff.real.mean(), diff.real.std())
print(diff.imag.mean(), diff.imag.std())