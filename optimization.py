import numpy as np
from machines import base
from utils import calc
from utils import optimizers
from typing import Any, Callable, List, Optional, Tuple

GradCalc = Callable[base.BaseMachine,
                    Tuple[np.ndarray, np.ndarray, float, List[float]]]


def exact(exact_state: np.ndarray, machine_type: Any, grad_func: GradCalc,
          n_epochs: int, n_message: Optional[int] = None,
          optimizer: Optional[optimizers.BaseOptimizer] = None):
  """Main optimization script for exact (deterministic) calculations.

  Args:
    exact_state: Exact state evolution with shape (T + 1, 2^N).
    machine_type: A class creator from the implemented machines.
      This will be used to create the machine to optimize.
      The class should inherit machines.BaseMachine
    grad_func: Method that calculates gradients.
      See `GradCalc` typing for the type of arguments and returns.
      Should be one of `energy.deterministic` gradient calculation methods
      with `ham` and the rest arguments specified.
  """
  # TODO: Complete docstring
  time_steps = len(exact_state) - 1

  # Initialize machine
  machine = machine_type(exact_state[0], time_steps)
  if optimizer is None:
    optimizer = optimizers.AdamComplex(machine.shape, dtype=machine.dtype)

  history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
  for epoch in range(n_epochs):
    Ok, Ok_star_Eloc, Eloc, _ = grad_func(machine)

    grad = Ok_star_Eloc - Ok.conj() * Eloc
    if grad.shape[1:] != machine.shape[1:]:
      grad = grad.reshape((time_steps,) + machine.shape[1:])

    machine.update(optimizer(grad, epoch))

    # TODO: Make dense property in machines
    full_psi = machine.dense()

    # TODO: Make this a loop with a dictionary key -> calc method
    # to avoid code repetition
    history["exact_Eloc"].append(Eloc)
    history["overlaps"].append(calc.overlap(full_psi, exact_state))
    history["avg_overlaps"].append(calc.averaged_overlap(full_psi, exact_state))

    # Print messages
    if n_message is not None and epoch % n_message == 0:
      print("\nEpoch {}".format(epoch))
      for k in history.keys():
        print(": ".join(k, history[k][-1]))

  return history, full_psi