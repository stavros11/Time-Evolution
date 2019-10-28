"""Main optimization script for TensorFlow/Keras models.
Works only for the TFIM model! Uses automatric gradients.
We use a different main script for TensorFlow to avoid importing it in other
scripts.
"""
import argparse
import functools
import numpy as np
import tensorflow as tf

from machines.autograd import factory
from optimization import deterministic_auto
from optimization import optimize
from utils import saving
from utils import tfim
from typing import Optional


parser = argparse.ArgumentParser()
# Directories
parser.add_argument("--data-dir", default="/home/stavros/DATA/ClockV3/",
                    type=str, help="Basic directory that data is saved.")
parser.add_argument("--save-name", default="all_states", type=str,
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
parser.add_argument("--machine-type", default="FullWavefunction",
                    type=str, help="Machine name defined in `autograd.py`.")
parser.add_argument("--n-epochs", default=10000, type=int,
                    help="Number of epochs to train for.")
parser.add_argument("--time-grow-epochs", default=0, type=int,
                    help="Number of epochs for each time step during time growing.")
parser.add_argument("--learning-rate", default=1e-3, type=float,
                    help="Adam optimizer learning rate.")
parser.add_argument("--n-message", default=500, type=int,
                    help="Every how many epochs to display messages.")

# Sampling params
# TODO: Fix sampling - it is currently not implemented for Keras machines
#parser.add_argument("--n-samples", default=0, type=int,
#                    help="Number of samples for quantities calculation.")
#parser.add_argument("--n-corr", default=1, type=int,
#                    help="Number of correlation moves in MCMC sampler.")
#parser.add_argument("--n-burn", default=10, type=int,
#                    help="Number of burn-in moves in MCMC sampler.")
#parser.add_argument("--sample-time", action="store_true",
#                    help="Whether to sample time or assume uniform distribution")


def main(n_sites: int, time_steps: int, t_final: float, h_ev: float,
         machine_type: str,
         data_dir: str = "/home/stavros/DATA/Clock",
         save_name: str = "allstates",
         n_epochs: int = 0,
         learning_rate: float = 1e-3,
         n_message: Optional[int] = None,
         h_init: Optional[float] = None,
         init_state: Optional[np.ndarray] = None,
         time_grow_epochs: int = 0):
  """Main optimization script.

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
  exact_state, _ = tfim.tfim_exact_evolution(n_sites, t_final, time_steps,
                                             h0=h_init, h=h_ev,
                                             init_state=init_state)

  # Set optimizer
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  # Set machine
  machine_time_steps = 1 if time_grow_epochs > 0 else time_steps
  machine = getattr(factory, machine_type)(n_sites=n_sites,
                                           time_steps=machine_time_steps,
                                           init_state=exact_state[0],
                                           optimizer=optimizer)

  ham = tfim.tfim_hamiltonian(n_sites, h=h_ev, pbc=True)
  ham2 = ham.dot(ham)
  ham_tf = tf.cast(ham, dtype=machine.output_type)
  ham2_tf = tf.cast(ham2, dtype=machine.output_type)

  opt_params = {"n_epochs": n_epochs, "n_message": n_message}
  opt_params["exact_state"] = exact_state
  opt_params["grad_func"] = functools.partial(deterministic_auto.gradient,
            ham=ham_tf, dt=dt, ham2=ham2_tf)

  print("\n\nInitializing training with {}.".format(machine.name))
  filename = "{}_{}_N{}M{}".format(save_name, machine.name, n_sites, time_steps)
  print("Results will be saved at {}.".format(filename))

  grow_history, history = {}, {}
  # Set optimization function
  global_optimizer = functools.partial(optimize.globally, **opt_params)
  # Grow in time
  if time_grow_epochs > 0:
    global_optimizer.keywords["n_epochs"] = time_grow_epochs
    grow_history, machine = optimize.grow(machine, time_steps, global_optimizer)
    global_optimizer.keywords["exact_state"] = exact_state

    # Reset `global_optimizer` keywords in case we want to continue with
    # global optimization after growing time
    global_optimizer.keywords["n_epochs"] = n_epochs
    global_optimizer.keywords["n_message"] = n_message
    # Disable gradient mask to update all time steps
    global_optimizer.keywords["index_to_update"] = None

  if n_epochs > 0:
    history, machine = global_optimizer(machine)

  # Save training histories and final wavefunction
  saving.save_histories(data_dir, filename, history)
  saving.save_dense_wavefunction(data_dir, filename, machine.dense)


if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))