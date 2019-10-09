import numpy as np
import tensorflow as tf
tf.enable_v2_behavior()
from optimization import deterministic
from optimization import deterministic_auto
from machines import autograd
from utils import tfim

# TODO: Remove this script before merging

n_sites = 6
time_steps = 20
h_ev = 0.5

class EmptyMachine:
  pass

ham = tfim.tfim_hamiltonian(n_sites, h=h_ev, pbc=True)
ham2 = ham.dot(ham)
dt = 1e-3
dtype = tf.float32

init_wavefunction = 0.05*(np.random.random([21, 64]) + 1j * np.random.random([21, 64]))
model_real = autograd.fullwv_model(init_wavefunction.real, dtype=dtype)
model_imag = autograd.fullwv_model(init_wavefunction.imag, dtype=dtype)
machine = autograd.BaseAutoGrad(model_real, model_imag,
                                n_sites=n_sites, time_steps=time_steps,
                                name="keras_fullwv")

# Test gradients
ham_tf = tf.convert_to_tensor(ham, dtype=machine.output_type)
ham2_tf = tf.convert_to_tensor(ham2, dtype=machine.output_type)
grad, Ok, Eloc, _ = deterministic_auto.gradient(machine, ham_tf, dt, ham2=ham2_tf)
grad = (grad[0].values.numpy() + 1j * grad[1].values.numpy()) / 2.0
grad = grad.reshape((time_steps + 1, 2**n_sites))

np_machine = EmptyMachine()
np_machine.dense = machine.dense
Ok, Ok_star_Eloc, Eloc_np, Eloc_terms = deterministic.gradient(np_machine, ham, dt, ham2=ham2)
grad_np = Ok_star_Eloc - Ok.conj() * Eloc


print(Eloc - Eloc_np.real)
print(grad[1:] - grad_np)
print((grad[1:] - grad_np).max())