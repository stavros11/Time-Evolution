"""Optimizes full wavefunction using numpy.

Uses all states to calculate gradients.
Currently only TFIM is supported.
"""
import argparse
import functools
import numpy as np
import optimization
from machines import factory
from utils import optimizers
from utils import saving
from utils import tfim
from typing import Optional


parser = argparse.ArgumentParser()
# Directories
parser.add_argument("--data-dir", default="/home/stavros/DATA/Clock/",
                    type=str, help="Basic directory that data is saved.")
parser.add_argument("--save_name", default="allstates", type=str,
                    help="Name to use for distinguish the saved training data.")

# System params
parser.add_argument("--n-sites", default=6, type=int,
                    help="Number of sites in the TFIM chain.")
parser.add_argument("--time-steps", default=20, type=int,
                    help="Number of time steps to evolve for. The initial "
                          "condition is not included in this.")
parser.add_argument("--t-final", default=1.0, type=float,
                    help="Duration of the evolution.")
parser.add_argument("--h-ev", default=0.5, type=float,
                    help="Field under which TFIM is evolved.")
parser.add_argument("--h-init", default=1.0, type=float,
                    help="Field under which TFIM is initialized.")
# TODO: Add a flag for giving an `init_state` instead of `h_init`.

# Training params
parser.add_argument("--machine-type", default="FullWavefunctionMachine",
                    type=str,
                    help="Machine name as is imported in machines.factory.")
parser.add_argument("--n-epochs", default=10000, type=int,
                    help="Number of epochs to train for.")
parser.add_argument("--learning-rate", default=None, type=float,
                    help="Adam optimizer learning rate.")
parser.add_argument("--n-message", default=500, type=int,
                    help="Every how many epochs to display messages.")


def main(n_sites: int, time_steps: int, t_final: float, h_ev: float,
         n_epochs: int,
         machine_type: str,
         data_dir: str = "/home/stavros/DATA/Clock",
         save_name: str = "allstates",
         learning_rate: Optional[float] = None,
         n_message: Optional[int] = None,
         h_init: Optional[float] = None,
         init_state: Optional[np.ndarray] = None):
  """Main optimization script for exact (deterministic) calculations.

  Args:
    See parser definitions

  Saves:
    An .h5 with training histories in the directory
      '{data_dir}/histories/{generated save name}.h5'
    An .npy with the final wavefunction as a dense array in the directory
      '{data_dir}/final_dense/{generated save name}.npy'
    where 'generated save name = {save_name}_{machine.name}_N{n_sites}
              M{time_steps}'.
  """
  # Initialize TFIM Hamiltonian and calculate exact evolution
  if ((h_init is not None and init_state is not None) or
      (h_init is None and init_state is None)):
    raise ValueError("Exactly one of `h_init` and `init_state` should "
                     "be specified.")
  t_grid = np.linspace(0, t_final, time_steps + 1)
  dt = t_grid[1] - t_grid[0]
  ham = tfim.tfim_hamiltonian(n_sites, h=h_ev, pbc=True)
  exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev,
                                             init_state=init_state)


  # Set Clock energy calculation function and machine
  if machine_type not in factory.machine_to_gradfunc:
    raise ValueError("Uknown machine type {}.".format(machine_type))

  machine = getattr(factory, machine_type)(exact_state[0], time_steps)
  ham2 = ham.dot(ham)
  grad_func = factory.machine_to_gradfunc[machine_type]
  grad_func = functools.partial(grad_func, ham=ham, dt=dt, ham2=ham2)

  # Set optimizer
  optimizer = None
  if learning_rate is not None:
    optimizer = optimizers.AdamComplex(alpha=learning_rate)

  # Optimize
  history, machine = optimization.exact(exact_state, machine, grad_func,
                                        n_epochs, n_message,
                                        optimizer=optimizer)


  # Save training histories and final wavefunction
  filename = "{}_{}_N{}M{}".format(save_name, machine.name, n_sites, time_steps)
  saving.save_histories(data_dir, filename, history)
  saving.save_dense_wavefunction(data_dir, filename, machine.dense())


if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))