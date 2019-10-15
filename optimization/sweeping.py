"""Optimization by sweeping through time.

During every step only the variational parameters of a specific time are
updated while the rest are kept constant.
Obviously for this to work, the ansatz has to have explicit time dependence.
"""
import itertools
import numpy as np
from scipy.sparse import linalg
from machines import base
from utils import optimizers
from typing import Callable, List, Optional, Tuple


def masked_optimizer(optimizer_list: List[optimizers.BaseOptimizer],
                     optimization_steps_per_time: int
                     ) -> Callable[[np.ndarray, int], np.ndarray]:
  # TODO: Implement both ways sweeping
  shape, dtype = optimizer_list[0].shape, optimizer_list[1].dtype
  time_steps = len(optimizer_list)

  mask = np.zeros(shape, dtype=dtype)
  current_time = 0 # ignores initial condition because masks and optimizers
  mask[current_time] = np.ones(shape[1:], dtype=dtype)
  optimizer = optimizer_list[current_time]
  # lists are numbered from 0.
  opt_steps = 0
  while True:
    yield lambda grad, epoch: optimizer(grad * mask, epoch)
    opt_steps += 1
    if opt_steps == optimization_steps_per_time:
      mask[current_time] = np.zeros(shape[1:], dtype=dtype)
      current_time += 1
      if current_time == time_steps:
        current_time = 0
      mask[current_time] = np.ones(shape[1:], dtype=dtype)
      optimizer = optimizer_list[current_time]
      opt_steps = 0


class Base:

  def __init__(self):
    # Flag that keeps track of whether we are sweeping forward (1->T) or
    # backward (T->1).
    self.going_forward = True
    raise NotImplementedError

  def switch_direction(self):
    """Switches direction of sweep (in time)."""
    self.going_forward = not(self.going_forward)

  def single_step(self, machine: base.BaseMachine, time_step: int
                  ) -> base.BaseMachine:
    raise NotImplementedError

  def __call__(self, machine: base.BaseMachine) -> base.BaseMachine:
    """Updates the ansatz at a single time step.

    Args:
      machine: Machine object that defines the ansatz.
      time_step: Time to update

    Returns:
      The machine after the update.
    """
    if self.going_forward:
      for time_step in range(1, machine.time_steps + 1):
        machine = self.single_step(machine, time_step)
    else:
      for time_step in range(machine.time_steps - 1, 0, -1):
        machine = self.single_step(machine, time_step)
    return machine


class ExactGMResSweep(Base):
  """Sweep for full wavefunction that exactly minimizes loss at every step.

  Minimization is done by solving A|x> = |a>/2.
  See notes for the derivation of this equation.
  The system is solved using the `GMRes` method from `scipy`.

  This also works for the MPS ansatz as follows:
    The MPS is transformed to its dense form, the minimization system
    is solved and the dense solution is truncated back to MPS using SVDs.
  """

  def __init__(self, ham: np.ndarray, dt: float, maxiter: Optional[int] = None):
    dtham = 1j * dt * ham
    identity = np.eye(ham.shape[0], dtype=dtham.dtype)
    # Calculate A matrix
    # multiply by 2 so that you don't have to divide the alpha ket
    self.alpha_mat = identity + dt**2 * ham.dot(ham) / 2.0
    # Calculate expanded "unitary"
    self.exp_u1 = identity - dtham
    self.exp_u1d = identity + dtham
    self.maxiter = maxiter
    # Flag that keeps track of whether we are sweeping forward (1->T) or
    # backward (T->1).
    self.going_forward = True

  @classmethod
  def initialize(cls, ham: np.ndarray, dt: float,
                 one_term_mode: bool = False,
                 maxiter: Optional[int] = None):
    """Constructor that selects between standard and one-term mode."""
    sweeper = cls(ham, dt, maxiter)
    if one_term_mode:
      sweeper.single_step = sweeper.single_step_oneterm
    else:
      sweeper.single_step = sweeper.single_step_twoterms
    return sweeper

  def single_step_twoterms(self, machine: base.BaseMachine, time_step: int
                           ) -> base.BaseMachine:
    full_psi = machine.dense
    # Calculate alpha vector (ket)
    alpha_vec = self.exp_u1.dot(full_psi[time_step - 1])
    if time_step < len(full_psi) - 1:
      alpha_vec += self.exp_u1d.dot(full_psi[time_step + 1])
      alpha_vec *= 0.5
    new_psi, _ = linalg.gmres(self.alpha_mat, alpha_vec,
                              x0=full_psi[time_step],
                              maxiter=self.maxiter)
    # Update machine
    machine.update_time_step(new_psi, time_step)
    return machine

  def single_step_oneterm(self, machine: base.BaseMachine, time_step: int
                          ) -> base.BaseMachine:
    full_psi = machine.dense
    # Calculate alpha vector (ket)
    if self.going_forward:
      alpha_vec = self.exp_u1.dot(full_psi[time_step - 1])
    else:
      if time_step == len(full_psi) - 1:
        raise ValueError("Cannot perform backward update for the final "
                         "time step.")
      alpha_vec = self.exp_u1d.dot(full_psi[time_step + 1])
    new_psi, _ = linalg.gmres(self.alpha_mat, alpha_vec,
                              x0=full_psi[time_step],
                              maxiter=self.maxiter)
    # Update machine
    machine.update_time_step(new_psi, time_step)
    return machine


class NormalizedSweep(Base):

  def __init__(self, ham: np.ndarray, dt: float, steps_per_time: int = 1000,
               optimizer: Optional[optimizers.BaseOptimizer] = None):
    self.going_forward = True
    dtham = 1j * dt * ham
    identity = np.eye(ham.shape[0], dtype=dtham.dtype)
    # Calculate A matrix
    # multiply by 2 so that you don't have to divide the alpha ket
    self.alpha_mat = identity + dt**2 * ham.dot(ham) / 2.0
    # Calculate expanded "unitary"
    self.exp_u1 = identity - dtham
    self.exp_u1d = identity + dtham
    self.steps_per_time = steps_per_time

    self.optimizer = optimizer
    n_sites = int(np.log2(len(ham)))
    self.all_spin_confs = np.array(list(
        itertools.product([1, -1], repeat=n_sites)))

  def optimization_step(self, machine: base.BaseMachine,
                        u_psi_prev: np.ndarray,
                        time_step: int,
                        epoch: int) -> Tuple[base.BaseMachine, float]:
    psi_t = machine.dense[time_step]
    if self.going_forward:
      psi_n = machine.dense[time_step - 1]
    else:
      psi_n = machine.dense[time_step + 1]
    psi2_t = np.abs(psi_t)**2

    norm_t = (psi2_t + np.abs(psi_n)**2).sum()
    alpha_psi_t = self.alpha_mat.dot(psi_t)
    energy_t = (psi_t.conj().dot(alpha_psi_t) -
                2 * psi_t.conj().dot(u_psi_prev).real) / norm_t

    weights = machine.gradient(self.all_spin_confs, time_step)
    shape = (len(self.all_spin_confs),) + machine.shape[1:]
    slicer = (slice(None),) + len(machine.shape[1:]) * (np.newaxis,)
    weights = weights.reshape(shape) * psi2_t[slicer]

    Ok = weights.sum(axis=0) / norm_t
    Ok_star_Eloc = ((alpha_psi_t - u_psi_prev)[slicer] *
                    weights.conj()).sum(axis=0) / norm_t
    grad = Ok_star_Eloc - Ok.conj() * energy_t

    machine.update_time_step(self.optimizer(grad, epoch), time_step,
                             replace=True)
    return machine, energy_t

  def single_step(self, machine: base.BaseMachine, time_step: int
                  ) -> base.BaseMachine:
    if self.optimizer is None:
      self.optimizer = optimizers.AdamComplex(machine.shape[1:], machine.dtype,
                                              alpha=1e-5)
    full_psi = machine.dense
    if self.going_forward:
      u_psi_prev = self.exp_u1.dot(full_psi[time_step - 1])
    else:
      u_psi_prev = self.exp_u1d.dot(full_psi[time_step + 1])

    #print("\nOptimizing time step {}...".format(time_step))
    for epoch in range(self.steps_per_time):
      machine, current_energy = self.optimization_step(machine, u_psi_prev,
                                                       time_step, epoch)
      #if epoch % (self.steps_per_time // 10) == 0:
      #print("Epoch: {}, Energy(t): {}".format(epoch, current_energy))
    return machine