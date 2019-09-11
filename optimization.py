import numpy as np
from machines import base
from samplers import samplers
from utils import calc
from utils import optimizers
from typing import Callable, List, Optional, Tuple

GradCalc = Callable[[base.BaseMachine],
                    Tuple[np.ndarray, np.ndarray, float, List[float]]]
SamplingGradCalc = Callable[[base.BaseMachine, np.ndarray, np.ndarray],
                            Tuple[np.ndarray, np.ndarray, float, List[float]]]


def exact(exact_state: np.ndarray, machine: base.BaseMachine,
          grad_func: GradCalc, n_epochs: int, n_message: Optional[int] = None,
          optimizer: Optional[optimizers.BaseOptimizer] = None):
  """Main optimization script for exact (deterministic) calculations.

  Args:
    exact_state: Exact state evolution with shape (T + 1, 2^N).
    machine: Machine object to optimize.
    grad_func: Method that calculates gradients.
      See `GradCalc` typing for the type of arguments and returns.
      Should be one of `energy.deterministic` gradient calculation methods
      with `ham` and the rest arguments specified.
    n_epochs: Number of epochs to optimize for.
    n_message: Every how many epochs to print messages during optimization.
      If `None` no messages are printed.
    optimizer: Optimizer object to use for updating the machine.
      If `None` default Adam optimizer is used.
  """
  time_steps = len(exact_state) - 1

  if optimizer is None:
    optimizer = optimizers.AdamComplex(machine.shape, dtype=machine.dtype)

  history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
  for epoch in range(n_epochs):
    Ok, Ok_star_Eloc, Eloc, _ = grad_func(machine)

    grad = Ok_star_Eloc - Ok.conj() * Eloc
    if grad.shape[1:] != machine.shape[1:]:
      grad = grad.reshape((time_steps,) + machine.shape[1:])

    machine.update(optimizer(grad, epoch))

    full_psi = machine.dense
    history["exact_Eloc"].append(Eloc)
    history["overlaps"].append(calc.overlap(full_psi, exact_state))
    history["avg_overlaps"].append(calc.averaged_overlap(full_psi, exact_state))

    # Print messages
    if n_message is not None and epoch % n_message == 0:
      print("\nEpoch {}".format(epoch))
      for k in history.keys():
        print("{}: {}".format(k, history[k][-1]))

  return history, machine



def sampling(exact_state: np.ndarray, machine: base.BaseMachine,
             sampler: samplers.Base, grad_func: SamplingGradCalc,
             detenergy_func: Callable[[np.ndarray], float],
             n_epochs: int, n_message: Optional[int] = None,
             optimizer: Optional[optimizers.BaseOptimizer] = None):
  """Main optimization script for exact (deterministic) calculations.

  Args:
    exact_state: Exact state evolution with shape (T + 1, 2^N).
    machine: Machine object to optimize.
    grad_func: Method that calculates gradients.
      See `SamplingGradCalc` typing for the type of arguments and returns.
      Should be `energy.sampling.gradient` with some args specified.
    detenergy_func: Function that calculates deterministic energy.
    n_epochs: Number of epochs to optimize for.
    n_message: Every how many epochs to print messages during optimization.
      If `None` no messages are printed.
    optimizer: Optimizer object to use for updating the machine.
      If `None` default Adam optimizer is used.
  """
  time_steps = len(exact_state) - 1

  if optimizer is None:
    optimizer = optimizers.AdamComplex(machine.shape, dtype=machine.dtype)

  history = {"overlaps" : [], "avg_overlaps": [],
             "exact_Eloc": [], "sampled_Eloc": []}
  for epoch in range(n_epochs):
    # Sample
    configs, times = sampler(machine.dense)

    # Calculate gradient
    Ok, Ok_star_Eloc, Eloc, _, _ = grad_func(machine, configs, times)

    grad = Ok_star_Eloc - Ok.conj() * Eloc
    if grad.shape[1:] != machine.shape[1:]:
      grad = grad.reshape((time_steps,) + machine.shape[1:])
    machine.update(optimizer(grad, epoch))

    full_psi = machine.dense
    # Calculate energy exactly (using all states) on the current machine state
    exact_Eloc, _ = detenergy_func(full_psi)
    exact_Eloc = np.array(exact_Eloc).sum()

    history["exact_Eloc"].append(Eloc)
    history["exact_Eloc"].append(exact_Eloc)
    history["overlaps"].append(calc.overlap(full_psi, exact_state))
    history["avg_overlaps"].append(calc.averaged_overlap(full_psi, exact_state))

    # Print messages
    if n_message is not None and epoch % n_message == 0:
      print("\nEpoch {}".format(epoch))
      for k in history.keys():
        print("{}: {}".format(k, history[k][-1]))

  return history, machine