import numpy as np
from machines import base
from optimization import sweeping
from samplers import samplers
from utils import calc
from utils import optimizers
from typing import Callable, Dict, List, Optional, Union, Tuple

# Typings of gradient calculation functions
GradCalc = Callable[[base.BaseMachine],
                    Tuple[np.ndarray, np.ndarray, float, List[float]]]
SamplingGradCalc = Callable[[base.BaseMachine, np.ndarray, np.ndarray],
                            Tuple[np.ndarray, np.ndarray, float, List[float]]]


def globally(exact_state: np.ndarray,
             machine: base.BaseMachine,
             n_epochs: int,
             grad_func: Union[GradCalc, SamplingGradCalc],
             sampler: Optional[samplers.Base] = None,
             detenergy_func: Optional[Callable[[np.ndarray], float]] = None,
             optimizer: Optional[optimizers.BaseOptimizer] = None,
             n_message: Optional[int] = None
             ) -> Tuple[Dict[str, List[float]], base.BaseMachine]:
  """Optimizes the Clock Hamiltonian for a machine globally.

  Globally means that all time steps are optimized in a single optimziation
  step.

  Args:
    exact_state: Exact state evolution with shape (T + 1, 2^N).
    machine: Machine object to optimize.
    grad_func: Method that calculates gradients.
      See typing for the type of arguments and returns.
      Should be one of the gradient calculation methods defined unter`energy`
      with `ham` and the rest arguments specified.
    sampler: Sampler to use for calculating `configs` and `times` samples.
      If None, deterministic calculation mode is assumed.
    detenergy_func: Function that calculates deterministic energy.
      This is relevant only for the sampling case.
    optimizer: Optimizer object to use for updating the machine.
      If `None` default Adam optimizer is used.
    n_message: Every how many epochs to print messages during optimization.
      If `None` no messages are printed.

  Returns:
    history: Dictionary with the history of quantities we track during training.
      See definition of `history` in method for a list of tracked quantities.
    machine. Machine object after its optimization is completed.
  """
  time_steps = len(exact_state) - 1

  if optimizer is None:
    optimizer = optimizers.AdamComplex(machine.shape, dtype=machine.dtype)

  history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
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
    grad = Ok_star_Eloc - Ok.conj() * Eloc
    if grad.shape[1:] != machine.shape[1:]:
      grad = grad.reshape((time_steps,) + machine.shape[1:])

    # Update machine
    machine.update(optimizer(grad, epoch))

    # Calculate histories
    full_psi = machine.dense
    history["overlaps"].append(calc.overlap(full_psi, exact_state))
    history["avg_overlaps"].append(calc.averaged_overlap(full_psi, exact_state))
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


def sweep(exact_state: np.ndarray,
          machine: base.BaseMachine,
          sweeper: sweeping.Base,
          n_epochs: int,
          detenergy_func: Optional[Callable[[np.ndarray], float]] = None,
          both_directions: bool = True,
          n_message: Optional[int] = None
          ) -> Tuple[Dict[str, List[float]], base.BaseMachine]:
  """Optimizes the Clock Hamiltonian for a machine by sweeping through time.

  Args:
    exact_state: Exact state evolution with shape (T + 1, 2^N).
    machine: Machine object to optimize.
    sweeper: Class that implements the sweeping algorithm.
      This class should be callable (should have `__call__` implemented)
      and should take the `machine` and the time step as integer and return
      the machine after optimizing it for `t=time_step`.
    n_epochs: Total number of full sweeps. A full sweep optiizes all time steps,
      that is from t=1 to t=T or reverse.
    detenergy_func: Function that calculates deterministic energy.
      This is used for history only - not optimization.
    both_directions: If True the sweeping is performed back and forth.
      All odd sweeps are 1->T and even sweeps are T-->1.
      If False then all sweeps are 1->T.
    n_message: Every how many epochs to print messages during optimization.
      If `None` no messages are printed.

  Returns:
    history: Dictionary with the history of quantities we track during training.
      See definition of `history` in method for a list of tracked quantities.
    machine. Machine object after its optimization is completed.
  """
  history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
  for epoch in range(n_epochs):
    machine = sweeper(machine)
    if both_directions:
      sweeper.switch_direction()

    # Calculate histories
    full_psi = machine.dense
    history["overlaps"].append(calc.overlap(full_psi, exact_state))
    history["avg_overlaps"].append(calc.averaged_overlap(full_psi, exact_state))
    exact_Eloc, _ = detenergy_func(full_psi)
    history["exact_Eloc"].append(np.array(exact_Eloc).sum())

    # Print messages
    if n_message is not None and epoch % n_message == 0:
      print("\nSweep {}".format(epoch))
      for k, val in history.items():
        print("{}: {}".format(k, val[-1]))

  return history, machine


def sweep_global_norm(exact_state: np.ndarray,
                      machine: base.BaseMachine,
                      n_epochs: int,
                      grad_func: Union[GradCalc, SamplingGradCalc],
                      sampler: Optional[samplers.Base] = None,
                      detenergy_func: Optional[Callable[[np.ndarray], float]] = None,
                      optimizer: Optional[optimizers.BaseOptimizer] = None,
                      n_message: Optional[int] = None
                      ) -> Tuple[Dict[str, List[float]], base.BaseMachine]:
  """Optimizes the Clock energy (correct normalization) by sweeping through time.

  Copies the functionality of `global` optimization but masks the gradient
  so that only one time step is updated at each epoch.
  """
  # FIXME: Code repetition from `global`.
  # TODO: Allow user to give optimizer
  time_steps = len(exact_state) - 1
  if optimizer is not None:
    raise NotImplementedError

  # Define a different optimizer for each time step
  optimizer_list = [optimizers.AdamComplex(machine.shape, dtype=machine.dtype)
                    for _ in range(time_steps)]
  masked_optimizer = sweeping.masked_optimizer(optimizer_list)

  history = {"overlaps" : [], "avg_overlaps": [], "exact_Eloc": []}
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
    grad = Ok_star_Eloc - Ok.conj() * Eloc
    if grad.shape[1:] != machine.shape[1:]:
      grad = grad.reshape((time_steps,) + machine.shape[1:])

    # Update machine
    machine.update(next(masked_optimizer)(grad, epoch))

    # Calculate histories
    full_psi = machine.dense
    history["overlaps"].append(calc.overlap(full_psi, exact_state))
    history["avg_overlaps"].append(calc.averaged_overlap(full_psi, exact_state))
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