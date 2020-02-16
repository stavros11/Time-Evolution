import argparse
import functools
import numpy as np
import tensorflow as tf
import clock
import machines
from utils import calc, tfim
from typing import Optional


parser = argparse.ArgumentParser()
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

# Machine params
parser.add_argument("--machine-type", default="FullWavefunction", type=str)
parser.add_argument("--n-hidden", default=None, type=int)

# Optimization
parser.add_argument("--n-epochs", default=0, type=int,
                    help="Number of epochs to train for.")
parser.add_argument("--n-message", default=500, type=int,
                    help="Every how many epochs to display messages.")



def main(n_sites: int, time_steps: int, t_final: float, h_init: float,
         h_ev: float, n_epochs: int, n_message: int,
         machine_type: str, n_hidden: Optional[int] = None):
  dt = t_final / time_steps
  t = np.linspace(0, t_final, time_steps + 1)
  assert dt == t[1] - t[0]

  # Define system
  ham = tfim.tfim_hamiltonian(n_sites, h=h_ev)
  exact_state, exact_obs = tfim.tfim_exact_evolution(n_sites, t_final,
                                                     time_steps,
                                                     h0=h_init, h=h_ev)

  # Define ansatz
  machine_params = {"initial_condition": exact_state[0],
                    "time_steps": time_steps,
                    "dtype": tf.float32}
  if n_hidden is not None:
    machine_params["n_hidden"] = n_hidden
  model = getattr(machines, machine_type)(**machine_params)

  ham_tf = tf.convert_to_tensor(ham, dtype=model.ctype)
  ham2_tf = tf.cast(tf.matmul(ham, ham), dtype=model.ctype)
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



if __name__ == '__main__':
  args = parser.parse_args()
  main(**vars(args))
