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
             index_to_update: Optional[int] = None,
             subset_time_steps: Optional[List[int]] = None,
             update_time_zero: bool = False
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

  if subset_time_steps is not None:
    machine_to_update = machine.subset(subset_time_steps, machine,
                                       update_time_zero)
  else:
    machine_to_update = machine

  for epoch in range(n_epochs):
    # Calculate VMC quantities with sampling or exactly
    if sampler is None:
      Ok, Ok_star_Eloc, Eloc, _ = grad_func(machine_to_update)
    else:
      configs, times = sampler(machine_to_update.dense)
      Ok, Ok_star_Eloc, Eloc, _, _ = grad_func(machine_to_update, configs, times)

    # Calculate gradients
    grad = Ok_star_Eloc - Ok.conj() * Eloc
    if not update_time_zero and sampler is None:
      grad = grad[1:]

    if index_to_update is not None:
      assert index_to_update < machine_to_update.time_steps
      new_grad = np.zeros_like(grad)
      new_grad[index_to_update] = np.copy(grad[index_to_update])
      grad = new_grad

    # Update machine
    machine_to_update.update(grad, epoch, update_time_zero)
    if subset_time_steps is not None:
      machine.set_parameters(machine_to_update.tensors, subset_time_steps)

    # Calculate histories
    full_psi = machine.dense
    if exact_state is not None:
      history["overlaps"].append(calc.overlap(full_psi, exact_state))
      history["avg_overlaps"].append(calc.averaged_overlap(
          full_psi, exact_state))

    if sampler is not None:
      history["sampled_Eloc"].append(Eloc)

    if detenergy_func is None:
      history["exact_Eloc"].append(Eloc)
    else:
      exact_Eloc, _ = detenergy_func(full_psi)
      history["exact_Eloc"].append(np.array(exact_Eloc).sum())

    # Print messages
    if n_message is not None and epoch % n_message == 0:
      print("\nEpoch {}".format(epoch))
      for k, val in history.items():
        print("{}: {}".format(k, val[-1]))

  return history, machine


def sweep(machine: base.BaseMachine, global_optimizer: Callable,
          n_sweeps: int, sweep_mode: Optional[str] = None
          ) -> Tuple[Dict[str, List[float]], base.BaseMachine]:
  """Optimizes the Clock Hamiltonian by sweeping in time.

  The first sweep grows in time.
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

  history = {"sweeping_exact_Eloc": []}
  if "exact_state" in global_optimizer.keywords:
    n = len(global_optimizer.keywords["exact_state"])
    assert n == machine.time_steps + 1
    history["sweeping_overlaps"] = []
    history["sweeping_avg_overlaps"] = []

  if sweep_mode is None:
    print("Performing full sweeps sweeping one direction.")
  elif sweep_mode == "binary":
    print("Performing binary sweeps sweeping both directions.")
  elif sweep_mode == "triple":
    print("Performing triple sweeps sweeping both directions.")
  else:
    raise ValueError("Unknown sweep mode {}.".format(sweep_mode))
  binary = sweep_mode == "binary"
  triple = sweep_mode == "triple"

  subset_time_steps = [0]
  for i in range(n_sweeps):
    print("\nSweep {} / {}".format(i + 1, n_sweeps))

    # Create correct time step iterator according to the sweeping mode
    time_iter = range(machine.time_steps)
    if (binary or triple) and i % 2 == 1:
      time_iter = range(machine.time_steps - 1, 0, -1)
    elif triple and i % 2 == 0 and i > 0:
      time_iter = range(1, machine.time_steps - 1)

    for time_step in time_iter:
      if binary:
        if i % 2: # sweeping backwards
          step_history, machine = global_optimizer(
            machine, subset_time_steps=[time_step, time_step + 1],
            index_to_update=0, update_time_zero=True)
        else: # sweeping forward
          step_history, machine = global_optimizer(
            machine, subset_time_steps=[time_step, time_step + 1],
            index_to_update=None, update_time_zero=False)

      elif triple:
        if i == 0:
          step_history, machine = global_optimizer(
            machine, subset_time_steps=[time_step, time_step + 1],
            index_to_update=None, update_time_zero=False)
        else:
          subset_time_steps = [time_step - 1, time_step, time_step + 1]
          step_history, machine = global_optimizer(
            machine, subset_time_steps=subset_time_steps,
            index_to_update=1, update_time_zero=False)

      else:
        if i > 0:
          subset_time_steps = None
        else:
          subset_time_steps.append(time_step + 1)

        step_history, machine = global_optimizer(
            machine, index_to_update=time_step,
            subset_time_steps=subset_time_steps)

      if i == 0 and time_step < machine.time_steps - 1:
        # Initialization of next step when growing in time
        machine.set_parameters(
            np.array([machine.tensors[time_step + 1]]), [time_step + 2])

      print("\nTime step: {}".format(time_step + 1))
      for k, val in step_history.items():
        history["sweeping_{}".format(k)].append(val)
        print("{}: {}".format(k, val[-1]))

  return history, machine