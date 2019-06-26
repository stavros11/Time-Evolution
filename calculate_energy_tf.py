"""Calculates Clock energies and gradients. TensorFlow implementation.

Requires the full wavefunction at each time step and uses all states to
calculate expectation values deterministically.
(exponentially expensive sum - no sampling!)
"""
import tensorflow as tf


def calculate_norm(full_psi):
  phi_phi = tf.reduce_sum(tf.square(tf.abs(full_psi)))  
  return tf.cast(phi_phi, dtype=full_psi.dtype)


def all_states_Heff(psi_all, Ham, dt, phi_phi=None, Ham2=None):
  """Calculates Clock's expectation value.
  
  Args:
    psi_all: Full wavefunction at each time step of shape (M+1, 2**N)
      where N is the number of sites and M+1 the number of time steps.
    Ham: Full (real space) Hamiltonian matrix of shape (2**N, 2**N).
    dt: Time step.
    Ham2: Square of spin Hamiltonian of shape (2**N, 2**N).
    phi_phi: Norm of the full wave function.
      
  If Ham2 or phi_phi are not given, they are calculated from Ham and psi_all.
  We allow the user to give them to speed up the calculation
  (avoid calculating them multiple times).
  
  Returns:
    Heff_samples: Array of shape (M+1, 2**N) with the Hamiltonian terms
      before the reduce sum.
  """
  M = len(psi_all) - 1
  
  if phi_phi is None:
    phi_phi = calculate_norm(psi_all)
  
  if Ham2 is None:
    Ham2 = tf.matmul(Ham, Ham)
  
  # H^0 term
  n_Heff_0 = psi_all[0] - psi_all[1]
  n_Heff_M = psi_all[M] - psi_all[M-1]
  n_Heff = 2 * psi_all[1:M] - psi_all[:M-1] - psi_all[2:]
    
  # H^1 term
  n_Heff_0 -= 1j * dt * tf.matmul(Ham, psi_all[1][:, tf.newaxis])[:, 0]
  n_Heff_M += 1j * dt * tf.matmul(Ham, psi_all[M-1][:, tf.newaxis])[:, 0]
  n_Heff += 1j * dt * tf.matmul(psi_all[:M - 1] - psi_all[2:], Ham, 
                                transpose_b=True)
  
  # H^2 term
  n_Heff2_all = tf.matmul(psi_all, Ham2, transpose_b=True)
  n_Heff_0 += 0.5 * dt * dt * n_Heff2_all[0]
  n_Heff_M += 0.5 * dt * dt * n_Heff2_all[M]
  n_Heff += dt * dt * n_Heff2_all[1:M]
    
  return tf.concat([n_Heff_0[tf.newaxis], n_Heff, n_Heff_M[tf.newaxis]], 
                   axis=0)
  
  
def all_states_Eloc(full_psi, Ham, dt, Ham2=None):
  """Calculates expectation value of the local energy."""
  if Ham2 is None:
    Ham2 = tf.matmul(Ham, Ham)
  
  norm = calculate_norm(full_psi)
  Heff_samples = all_states_Heff(full_psi, Ham, dt, phi_phi=norm, Ham2=Ham2)
  return tf.reduce_sum((tf.conj(full_psi) * Heff_samples)) / norm


def all_states_gradient(full_psi, Ham, dt, Ham2=None):
  """Gradients of the Clock Hamiltonian with respect to a full wavefunction.
  
  Args:
    full_psi: Full wavefunction at each time step of shape (M + 1, 2**N)
      where N is the number of sites and M+1 the number of time steps.
    Ham: Full (real space) Hamiltonian matrix of shape (2**N, 2**N).
    dt: Time step.
    Ham2: Square of spin Hamiltonian of shape (2**N, 2**N).
      If Ham2 is not given then it is calculated from Ham.

  Returns:
    Ok: EV of gradient of Clock with respect to every wavefunction component.
    Ok_star_Eloc: EV of Ok*Eloc term with respect to every wavefunc. component.
    Eloc: EV of Eloc (number).
      
  Ok and Ok_star_Eloc have shape (M, 2**N) (or (M + 1, 2**N) if psi0 is given).
  """
  M, Nstates = full_psi.shape
  M += -1
  
  if Ham2 is None:
    Ham2 = tf.matmul(Ham, Ham)
  
  phi_phi = calculate_norm(full_psi)
  Heff_samples = all_states_Heff(full_psi, Ham, dt, phi_phi=phi_phi, Ham2=Ham2)

  Eloc = tf.reduce_sum((tf.conj(full_psi) * Heff_samples)) / phi_phi

  # shape (M, Nstates)
  Ok = tf.conj(full_psi)[1:] / phi_phi
  Ok_star_Eloc = Heff_samples[1:] / phi_phi

  return Ok, Ok_star_Eloc, Eloc
