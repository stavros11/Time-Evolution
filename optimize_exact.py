"""Optimizes full wavefunction using numpy.

Uses all states to calculate gradients.
"""
import functools
import h5py
import numpy as np
import optimization
from energy import deterministic
from utils import optimizers
from utils import tfim
from typing import Any, Optional


n_sites = 6
time_steps = 20
t_final = 1.0
h_init = 1.0
h_ev = 0.5
n_epochs = 10000
n_message = 500
init_prod = False


def main(n_sites: int, time_steps: int, t_final: float, machine_type: Any,
         h_ev: float, run_name: str, n_epochs: int,
         learning_rate: Optional[float] = None,
         n_message: Optional[int] = None,
         h_init: Optional[float] = None,
         init_state: Optional[np.ndarray] = None):
  """Main optimization script for exact (deterministic) calculations."""
  # TODO: Complete docstring

  # Initialize TFIM Hamiltonian and calculate exact evolution
  if ((h_init is not None and init_state is not None) or
      (h_init is None and init_state is None)):
    raise ValueError("Exactly one of `h_init` and `init_state` should "
                     "be specified.")

  ham = tfim.tfim_hamiltonian(n_sites, h=h_ev, pbc=True)
  exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev,
                                             init_state=init_state)


  # Prepare Clock energy calculation function
  if machine_type not in deterministic.machine_to_gradfunc:
    raise ValueError("Uknown machine type {}.".format(machine_type))
  ham2 = ham.dot(ham)
  grad_func = deterministic.machine_to_gradfunc[machine_type]
  grad_func = functools.partial(grad_func, ham=ham, ham2=ham2)

  # Set optimizer
  optimizer = None
  if learning_rate is not None:
    optimizer = optimizers.AdamComplex(alpha=learning_rate)


  # Optimize
  # TODO: Change `machine_type` to a machine object that you create here
  # to make it more readable
  history, full_psi = optimization.exact(exact_state, machine_type, grad_func,
                                         n_epochs, n_message,
                                         optimizer=optimizer)


  # TODO: Fix filenames
  # Save history
  if init_prod:
    filename = "al{}_N{}M{}.h5py".format(machine.name, n_sites, time_steps)
  else:
    filename = "allstates_{}_N{}M{}.h5py".format(machine.name, n_sites, time_steps)
  file = h5py.File("histories/{}".format(filename), "w")
  for k in history.keys():
    file[k] = history[k]
  file.close()

  # Save final dense wavefunction
  filename = "{}.npy".format(filename[:-5])
  np.save("final_dense/{}".format(filename), full_psi)
