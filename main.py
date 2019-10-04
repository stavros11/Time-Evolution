"""Main optimization script.

Works only for the TFIM model!
"""
import argparse
import functools
import numpy as np
from optimization import deterministic
from optimization import optimize
from optimization import sampling
from machines import factory
from samplers import samplers
from utils import optimizers
from utils import saving
from utils import tfim
from typing import Optional

# TODO: Update README


parser = argparse.ArgumentParser()
# Directories
parser.add_argument("--data-dir", default="/home/stavros/DATA/ClockV3/",
                    type=str, help="Basic directory that data is saved.")
parser.add_argument("--save-name", default="allstates", type=str,
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
                    type=str,
                    help="Machine name as is imported in machines.factory.")
parser.add_argument("--n-epochs", default=10000, type=int,
                    help="Number of epochs to train for.")
parser.add_argument("--learning-rate", default=None, type=float,
                    help="Adam optimizer learning rate.")
parser.add_argument("--n-message", default=500, type=int,
                    help="Every how many epochs to display messages.")

# Sweeping parms
parser.add_argument("--sweep-opt", action="store_true",
                    help="Optimize by sweeping through time.")
parser.add_argument("--sweep-both-directions", action="store_true",
                    help="Optimize by sweeping through time.")
parser.add_argument("--sweep-with-one-term", action="store_true",
                    help="Optimize by sweeping through time.")

# Sampling params
parser.add_argument("--n-samples", default=0, type=int,
                    help="Number of samples for quantities calculation.")
parser.add_argument("--n-corr", default=1, type=int,
                    help="Number of correlation moves in MCMC sampler.")
parser.add_argument("--n-burn", default=10, type=int,
                    help="Number of burn-in moves in MCMC sampler.")
parser.add_argument("--sample-time", action="store_true",
                    help="Whether to sample time or assume uniform distribution")

# Additional machine parameters
parser.add_argument("--d-bond", default=None, type=int,
                    help="MPS bond dimension (relevant for MPS machines).")
parser.add_argument("--d-phys", default=None, type=int,
                    help="MPS physical dimension (relevant for MPS machines).")


def main(n_sites: int, time_steps: int, t_final: float, h_ev: float,
         n_epochs: int,
         machine_type: str,
         n_samples: int, n_corr: int = 1, n_burn: int = 10,
         sample_time: bool = True,
         data_dir: str = "/home/stavros/DATA/Clock",
         save_name: str = "allstates",
         sweep_opt: bool = False,
         sweep_both_directions: bool = True,
         sweep_with_one_term: bool = False,
         learning_rate: Optional[float] = None,
         n_message: Optional[int] = None,
         h_init: Optional[float] = None,
         init_state: Optional[np.ndarray] = None,
         **machine_params):
  """Main optimization script.

  If n_samples == 0, deterministic calculation is used and all other sampling
  parameters are ignored. Otherwise quantities are calculated with sampling.

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

  # Set machine
  machine_params = {k: p for k, p in machine_params.items() if p is not None}
  machine = getattr(factory, machine_type)(exact_state[0], time_steps,
                    **machine_params)

  ham2 = ham.dot(ham)
  opt_params = {"exact_state": exact_state, "machine": machine,
                "n_epochs": n_epochs, "n_message": n_message}
  if sweep_opt:
    opt_params["both_directions"] = sweep_both_directions
    sweeper_type = factory.machine_to_sweeper[machine_type]
    # TODO: Add `maxiter` flag
    opt_params["sweeper"] = sweeper_type.initialize(ham, dt,
              one_term_mode=sweep_with_one_term)
    opt_params["detenergy_func"] = functools.partial(deterministic.energy,
                                                     ham=ham, dt=dt,
                                                     ham2=ham2)
    # Optimize
    history, machine = optimize.sweep(**opt_params)

  else:
    # Set optimizer
    optimizer = None
    if learning_rate is not None:
      optimizer = optimizers.AdamComplex(machine.shape, dtype=machine.dtype,
                                         alpha=learning_rate)

      opt_params["optimizer"] = optimizer

    # Set gradient and deterministic energy calculation functions
    if n_samples > 0:
      opt_params["grad_func"] = functools.partial(sampling.gradient,
                                                  dt=dt, h=h_ev)
      opt_params["detenergy_func"] = functools.partial(deterministic.energy,
                                                       ham=ham, dt=dt,
                                                       ham2=ham2)
      # Initialize sampler
      sampler = [samplers.SpinOnly, samplers.SpinTime][sample_time]
      opt_params["sampler"] = sampler(n_sites, time_steps, n_samples,
                                      n_corr, n_burn)
    else:
      opt_params["grad_func"] = functools.partial(
          machine.deterministic_gradient_func, ham=ham, dt=dt, ham2=ham2)

    # Optimize
    history, machine = optimize.globally(**opt_params)

  # Save training histories and final wavefunction
  filename = "{}_{}_N{}M{}".format(save_name, machine.name, n_sites, time_steps)
  saving.save_histories(data_dir, filename, history)
  saving.save_dense_wavefunction(data_dir, filename, machine.dense)


if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))