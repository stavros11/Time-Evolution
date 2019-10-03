"""Optimization by sweeping through time.

During every step only the variational parameters of a specific time are
updated while the rest are kept constant.
Obviously for this to work, the ansatz has to have explicit time dependence.
"""
import numpy as np
from scipy.sparse import linalg
from machines import base
from typing import Optional


class Base:

  def __init__(self):
    raise NotImplementedError

  def __call__(self, machine: base.BaseMachine, time_step: int
               ) -> base.BaseMachine:
    """Updates the ansatz at a single time step.

    Args:
      machine: Machine object that defines the ansatz.
      time_step: Time to update

    Returns:
      The machine after the update.
    """
    raise NotImplementedError


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

  def switch_direction(self):
    """Switches direction of sweep (in time)."""
    # TODO: Move this in `Base`.
    self.going_forward = not(self.going_forward)

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

  def __call__(self, machine: base.BaseMachine) -> base.BaseMachine:
    if self.going_forward:
      for time_step in range(1, machine.time_steps + 1):
        machine = self.single_step(machine, time_step)
    else:
      for time_step in range(machine.time_steps - 1, 0, -1):
        machine = self.single_step(machine, time_step)
    return machine