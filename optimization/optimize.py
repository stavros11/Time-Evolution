import numpy as np
from machines import base
from samplers import samplers
from utils import calc
from typing import Callable, Dict, List, Optional, Union, Tuple

# Typings of gradient calculation functions
GradCalc = Callable[[base.BaseMachine],
                    Tuple[np.ndarray, np.ndarray, float, List[float]]]
SamplingGradCalc = Callable[[base.BaseMachine, np.ndarray, np.ndarray],
                            Tuple[np.ndarray, np.ndarray, float, List[float]]]


def globally(machine: base.BaseMachine,
             n_epochs: int,
             grad_func: Union[GradCalc, SamplingGradCalc],
             sampler: Optional[samplers.Base] = None,
             exact_state: Optional[np.ndarray] = None,
             detenergy_func: Optional[Callable[[np.ndarray], float]] = None,
             n_message: Optional[int] = None,
             index_to_update: Optional[int] = None
             ) -> Tuple[Dict[str, List[float]], base.BaseMachine]:
  """Optimizes the Clock Hamiltonian for a machine globally.

  Globally means that all time steps are optimized in a single optimziation
  step.

  Args:
    machine: Machine object to optimize.
    grad_func: Method that calculates gradients.
      See typing for the type of arguments and returns.
      Should be one of the gradient calculation methods defined unter`energy`
      with `ham` and the rest arguments specified.
    sampler: Sampler to use for calculating `configs` and `times` samples.
      If None, deterministic calculation mode is assumed.
    exact_state: Exact state evolution with shape (T + 1, 2^N).
    detenergy_func: Function that calculates deterministic energy.
      This is relevant only for the sampling case.
    n_message: Every how many epochs to print messages during optimization.
      If `None` no messages are printed.
    index_to_update: If this is not `None` then the gradient is masked so
      that only this index is updated.

  Returns:
    history: Dictionary with the history of quantities we track during training.
      See definition of `history` in method for a list of tracked quantities.
    machine: Machine object after its optimization is completed.
  """
  history = {"exact_Eloc": []}
  if exact_state is not None:
    assert len(exact_state) == machine.time_steps + 1
    history["overlaps"] = []
    history["avg_overlaps"] = []

  if sampler is not None:
    history["sampled_Eloc"] = []
    if detenergy_func is None:
      raise ValueError("Sampler was given but `detenergy_func` not.")

  for epoch in range(n_epochs):
    # Calculate VMC quantities with sampling or exactly
    if sampler is None:
      Ok, Ok_star_Eloc, Eloc, _ = grad_func(machine)
    else:
      configs, times = sampler(machine.dense)
      Ok, Ok_star_Eloc, Eloc, _, _ = grad_func(machine, configs, times)

    # Calculate gradients
    if Ok_star_Eloc is None:
      # this means that we are using automatic gradients and we can directly
      # update the machine using Ok.
      machine.update(Ok)
    else:
      # in this case we have to use VMC formulas for the gradients and do
      # the update in numpy.
      grad = Ok_star_Eloc - Ok.conj() * Eloc
      if index_to_update is not None:
        assert index_to_update < machine.time_steps
        new_grad = np.zeros_like(grad)
        new_grad[index_to_update] = np.copy(grad[index_to_update])
        grad = new_grad
      # Update machine
      machine.update(grad, epoch)

    # Calculate histories
    full_psi = machine.dense
    if exact_state is not None:
      history["overlaps"].append(calc.overlap(full_psi, exact_state))
      history["avg_overlaps"].append(calc.averaged_overlap(
          full_psi, exact_state))

    if detenergy_func is not None:
      history["sampled_Eloc"] = Eloc
      # Calculate energy exactly (using all states) on the current machine state
      exact_Eloc, _ = detenergy_func(full_psi)
      history["exact_Eloc"].append(np.array(exact_Eloc).sum())
    else:
      history["exact_Eloc"].append(Eloc)

    # Print messages
    if n_message is not None and epoch % n_message == 0:
      print("\nEpoch {}".format(epoch))
      for k, val in history.items():
        print("{}: {}".format(k, val[-1]))

  return history, machine


def grow(machine: base.BaseMachine, time_steps: int, global_optimizer: Callable
         ) -> Tuple[Dict[str, List[float]], base.BaseMachine]:
  """Optimizes the Clock Hamiltonian by sweeping and growing in time.

  This is typically used as a first pass before sweeping or global optimization.
  Note that messages are printed after every time step is optimized and
  currently user cannot change that.
  (there is no `n_message` flag for this method)

  Args:
    machine: Machine object to optimize.
      The given machine should only have one time step (+ initial condition)
      as the additional time steps will be added from the current method
      as we optimize.
    time_steps: Final number of time steps we want the machine to reach.
    global_optimizer: This is a partial function of `optimize.globally`.
      Should have all the arguments except `machine` defined.
      Some of these arguments are altered bellow.

  Returns:
    history: Dictionary with the history of quantities we track during training.
      See definition of `history` in method for a list of tracked quantities.
    machine: Machine object after its optimization is completed.
  """
  if "exact_state" in global_optimizer.keywords:
    exact_state = np.copy(global_optimizer.keywords["exact_state"])
  else:
    exact_state = None

  history = {"growing_exact_Eloc": []}
  if exact_state is not None:
    assert len(exact_state) == time_steps + 1
    history["growing_overlaps"] = []
    history["growing_avg_overlaps"] = []

  # Disable global optimization messages during growing in time as we will
  # print messages from this function
  if "n_message" in global_optimizer.keywords:
    global_optimizer.keywords["n_message"] = None

  # Grow in time
  for time_step in range(time_steps):
    if time_step > 0:
      machine.add_time_step()

    print("\nOptimizing time step {}...".format(time_step + 1))
    # Mask global gradient to update only one time step
    global_optimizer.keywords["index_to_update"] = time_step
    if exact_state is not None:
      global_optimizer.keywords["exact_state"] = exact_state[:time_step + 2]
    step_history, machine = global_optimizer(machine)

    for k, val in step_history.items():
      history["growing_{}".format(k)].append(val)
      print("{}: {}".format(k, val[-1]))

  return history, machine


def sweep(machine: base.BaseMachine, global_optimizer: Callable,
          n_sweeps: int, both_directions: bool = False
          ) -> Tuple[Dict[str, List[float]], base.BaseMachine]:
  """Optimizes the Clock Hamiltonian by sweeping in time.

  Note that messages are printed after every time step is optimized and
  currently user cannot change that.
  (there is no `n_message` flag for this method)

  Args:
    machine: Machine object to optimize.
    global_optimizer: This is a partial function of `optimize.globally`.
      Should have all the arguments except `machine` defined.
      Some of these arguments are altered bellow.
    n_sweeps: Total number of sweeps to perform.
    both_directions: If False only 1 --> T sweeps are performed. Otherwise
      we alternate between this and T --> 1 sweeps.
      In the latter case we start with T --> 1 sweep because it is assumed that
      this method is used after `grow` that performs the first 1 -- > T sweep.

  Returns:
    history: Dictionary with the history of quantities we track during training.
      See definition of `history` in method for a list of tracked quantities.
    machine: Machine object after its optimization is completed.
  """
  if n_sweeps <= 0:
    return dict(), machine

  if "exact_state" in global_optimizer.keywords:
    exact_state = global_optimizer.keywords["exact_state"]
  else:
    exact_state = None

  history = {"sweeping_exact_Eloc": []}
  if exact_state is not None:
    assert len(exact_state) == machine.time_steps + 1
    history["sweeping_overlaps"] = []
    history["sweeping_avg_overlaps"] = []

  for i in range(n_sweeps):
    if both_directions and i % 2 == 0:
      time_iter = range(machine.time_steps - 1, -1, -1)
    else:
      time_iter = range(machine.time_steps)

    for time_step in time_iter:
      global_optimizer.keywords["index_to_update"] = time_step
      step_history, machine = global_optimizer(machine)

      print("\nTime step: {}".format(time_step + 1))
      for k, val in step_history.items():
        history["sweeping_{}".format(k)].append(val)
        print("{}: {}".format(k, val[-1]))

  return history, machine
