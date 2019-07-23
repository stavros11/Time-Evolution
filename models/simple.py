"""Simple models. See base.py for method documentation."""

import numpy as np
import itertools
import tensorflow as tf
import utils
from models import base
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda


class FullStateModelAutograd(base.BaseModel):
  """Variational parameters are the full wavefunction as each time step."""

  def __init__(self, init_state, time_steps, std=1e-3,
               rtype=tf.float32, ctype=tf.complex64):
    state = np.array(time_steps * [init_state])
    state += utils.random_normal_complex(shape=state.shape, std=std)
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)
    self.real = tf.Variable(state.real, dtype=rtype, trainable=True)
    self.imag = tf.Variable(state.imag, dtype=rtype, trainable=True)

    self.vars = [self.real, self.imag]

  def variational_wavefunction(self, training=False):
    return tf.complex(self.real, self.imag)


class FullStateModel(FullStateModelAutograd):
  """Same as above but with hard-coded gradients."""

  def update(self, optimizer, updater, normalize=False):
    """Updates the state using hard-coded gradients.

    Args:
      optimizer: TensorFlow optimizer object to be used for the update.
      updater: Function that returns (complex_gradients, Eloc) tuple.
    """
    complex_grads, Eloc = updater(self.wavefunction())
    grads = [tf.real(complex_grads), tf.imag(complex_grads)]
    variables = [self.real, self.imag]
    optimizer.apply_gradients(zip(grads, variables))

    if normalize:
      norm = tf.sqrt(tf.reduce_sum(
          tf.square(tf.abs(self.variational_wavefunction())), axis=1))
      self.real.assign(self.real / norm[:, tf.newaxis])
      self.imag.assign(self.imag / norm[:, tf.newaxis])
    return Eloc


class MPSModel(base.BaseModel):
  """Matrix product state model.

  We assume small systems in the BaseModel, therefore calculations here
  are done naively by getting the dense wavefunction. For MPS it is possible
  to calculate quantities exactly even for larger systems by contracting,
  however this is not implemented here.

  Currently works only when n_sites is a power of 2.
  """

  def __init__(self, init_state, time_steps, d_bond, d_phys=2, std=1e-3,
               rtype=tf.float32, ctype=tf.complex64):
    self.n_sites = int(np.log2(len(init_state)))
    self.time_steps = time_steps
    self.d_bond, self.d_phys = d_bond, d_phys
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)

    #shape = (self.n_sites, time_steps, d_phys, d_bond, d_bond)
    #tensors = np.array(np.prod(shape[:3]) * [np.eye(d_bond)]).astype(np.complex128)
    #tensors = tensors.reshape(shape)
    #tensors += utils.random_normal_complex(shape=shape, std=std)
    tensors = np.array(time_steps * [self._dense_to_mps(init_state)])
    tensors = tensors.transpose([1, 0, 3, 2, 4])

    self.real = tf.Variable(tensors.real, dtype=rtype, trainable=True)
    self.imag = tf.Variable(tensors.imag, dtype=rtype, trainable=True)
    self.vars = [self.real, self.imag]

  def _dense_to_mps(self, state):
    """Transforms a dense wavefunction to an approximate MPS form."""
    tensors = [state[np.newaxis, :, np.newaxis]]
    while len(tensors) < self.n_sites:
      tensors = [m for t in tensors for m in self._svd_split(t)]

    array = np.zeros([self.n_sites, self.d_bond, self.d_phys, self.d_bond], dtype=state.dtype)
    for i in range(self.n_sites):
      array[i, :tensors[i].shape[0], :, :tensors[i].shape[-1]] = tensors[i][:self.d_bond, :, :self.d_bond]
    return array

  def _svd_split(self, m):
    """Splits an MPS tensor in two MPS tensors.

    Args:
      m: MPS tensor with shape (Dl, d, Dr)

    Returns:
      ml: Left tensor after split with shape (Dl, d', Dm)
      mr: Right tensor after split with shape (Dm, d', Dr)
      with d' = sqrt(d) and Dm = d' min(Dl, Dr)
    """
    Dl, d, Dr = m.shape
    d_new = int(np.sqrt(d))
    u, s, v = np.linalg.svd(m.reshape(Dl * d_new, Dr * d_new))
    d_middle = min(u.shape[-1], v.shape[0])
    s = np.diag(s[:d_middle])

    u = u[:, :d_middle].reshape((Dl, d_new, d_middle))
    sv = s.dot(v[:d_middle]).reshape((d_middle, d_new, Dr))
    return u, sv

  def _contract_tensors(self, tensors):
    n = int(tensors.shape[0])
    d = self.d_phys
    while n > 1:
      n = n // 2
      d *= d
      tensors = tf.einsum("italm,itbmr->itablr", tensors[::2], tensors[1::2])
      tensors = tf.reshape(tensors,
                           (n, self.time_steps, d, self.d_bond, self.d_bond))
    return tf.linalg.trace(tensors[0])

  def variational_wavefunction(self, training=False):
    """Calculates the dense form of MPS."""
    return self._contract_tensors(tf.complex(self.real, self.imag))



class RBMModel(base.BaseModel):

  def __init__(self, init_state, time_steps, n_hidden, std=1e-3,
               rtype=tf.float32, ctype=tf.complex64,
               init_dir="rbm_init_weights"):
    n_sites = int(np.log2(len(init_state)))
    self.time_steps = time_steps
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)

    import os
    norm_dir = "rbm_n{}h{}_w_norm.npy".format(n_sites, n_hidden)
    w_norm = np.load(os.path.join(os.getcwd(), init_dir, norm_dir))
    phase_dir = "rbm_n{}h{}_w_phase.npy".format(n_sites, n_hidden)
    w_phase = np.load(os.path.join(os.getcwd(), init_dir, phase_dir))

    w_norm = self._initialize_time_weights(w_norm, time_steps, std)
    self.w_norm = tf.Variable(w_norm, dtype=rtype, trainable=True)
    w_phase = self._initialize_time_weights(w_phase, time_steps, std)
    self.w_phase = tf.Variable(w_phase, dtype=rtype, trainable=True)
    self.vars = [self.w_norm, self.w_phase]

    all_states = np.array(list(itertools.product([-1, 1], repeat=n_sites)))
    self.all_states = tf.cast(all_states, dtype=rtype)

  @staticmethod
  def _initialize_time_weights(w0, time_steps, std=1e-3):
    w = np.array(time_steps * [w0])
    return w + np.random.normal(0.0, std, size=w.shape)

  @staticmethod
  def _time_logrbm_forward(w, x):
    """Forward prop of RBM where w has a time index in 0th axis.

    Einsum notation: t: Time, h: hidden, v: visible, b: batch.
    """
    w_x = tf.einsum("thv,bv->thb", w, x)
    return tf.reduce_sum(tf.cosh(w_x), axis=1)

  def variational_wavefunction(self, training=False):
    logpsi_re = self._time_logrbm_forward(self.w_norm, self.all_states)
    logpsi_im = tf.exp(self._time_logrbm_forward(self.w_phase, self.all_states))
    return tf.exp(tf.complex(logpsi_re, logpsi_im))


class SequentialDenseModel(base.BaseModel):
  """Simple sequential model with layers defined in _create_model method.

  Takes as input a spin configuration of shape (N,) and outputs the amplitudes
  for each time step with shape (M,) where M = time_steps. Two distinct NNs
  are used for the amplitude norms and the complex phases.
  TF autograd is used for training.
  """

  def __init__(self, init_state, time_steps,
               rtype=tf.float32, ctype=tf.complex64):
    n_sites = int(np.log2(len(init_state)))
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)

    self.all_confs = np.array(list(itertools.product([0, 1], repeat=n_sites)))
    self.all_confs = tf.cast(self.all_confs, dtype=rtype)

    self.model_norm = self._create_model(n_sites, time_steps)
    self.model_phase = self._create_model(n_sites, time_steps)
    self.vars = (self.model_norm.trainable_variables +
                 self.model_phase.trainable_variables)

  @staticmethod
  def _create_model(d_in, d_out):
    return tf.keras.Sequential([Input(shape=(d_in,)),
                                Dense(d_out // 2),
                                Dense(d_out)])

  def variational_wavefunction(self, training=False):
    norm = self.model_norm(self.all_confs, training=training)
    phase = self.model_phase(self.all_confs, training=training)
    psi = tf.complex(norm * tf.cos(2 * np.pi * phase),
                     norm * tf.sin(2 * np.pi * phase))
    return tf.transpose(psi, [1, 0])