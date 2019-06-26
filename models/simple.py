"""Simple models. See base.py for method documentation."""

import numpy as np
import itertools
import tensorflow as tf
import utils
from models import base
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda


class FullStateModel(base.BaseModel):
  """Variational parameters are the full wavefunction as each time step."""

  def __init__(self, init_state, time_steps, std=1e-3,
               rtype=tf.float32, ctype=tf.complex64):
    state = np.array(time_steps * [init_state])
    state += utils.random_normal_complex(shape=state.shape, std=std)
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)
    self.real = tf.Variable(state.real, dtype=rtype, trainable=True)
    self.imag = tf.Variable(state.imag, dtype=rtype, trainable=True)

  def update(self, optimizer, updater):
    complex_grads = updater(self.wavefunction())
    grads = [tf.real(complex_grads), tf.imag(complex_grads)]
    variables = [self.real, self.imag]
    optimizer.apply_gradients(zip(grads, variables))

  def variational_wavefunction(self, training=False):
    return tf.complex(self.real, self.imag)


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

  def variational_wavefunction(self, training=False):
    """Calculates the dense form of MPS."""
    tensors = tf.complex(self.real, self.imag)
    n = int(tensors.shape[0])
    d = self.d_phys
    while n > 1:
      n = n // 2
      d *= d
      tensors = tf.einsum("italm,itbmr->itablr", tensors[::2], tensors[1::2])
      tensors = tf.reshape(tensors,
                           (n, self.time_steps, d, self.d_bond, self.d_bond))
    return tf.linalg.trace(tensors[0])


class RBMModel(base.BaseModel):

  def __init__(self, init_state, time_steps, n_hidden, std=1e-3,
               rtype=tf.float32, ctype=tf.complex64):
    n_sites = int(np.log2(len(init_state)))
    self.time_steps = time_steps
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)

    self.wre = tf.Variable(np.random.normal(0.0, std, size=[time_steps, n_hidden, n_sites]),
                           dtype=rtype, trainable=True)
    self.wim = tf.Variable(np.random.normal(0.0, std, size=[time_steps, n_hidden, n_sites]),
                           dtype=rtype, trainable=True)
    self.bre = tf.Variable(np.zeros((time_steps, n_hidden)),
                           dtype=rtype, trainable=True)
    self.bim = tf.Variable(np.zeros((time_steps, n_hidden)),
                           dtype=rtype, trainable=True)
    self.vars = [self.wre, self.wim, self.bre, self.bim]

    all_states = np.array(list(itertools.product([-1, 1], repeat=n_sites)))
    self.all_states = tf.cast(all_states, dtype=rtype)

  def variational_wavefunction(self, training=False):
    w_sigma_re = tf.einsum("thj,sj->ths", self.wre, self.all_states)
    w_sigma_im = tf.einsum("thj,sj->ths", self.wim, self.all_states)
    log_re = tf.log(tf.cosh(w_sigma_re + self.bre[:, :, tf.newaxis]))
    log_im = tf.log(tf.cosh(w_sigma_im + self.bim[:, :, tf.newaxis]))
    log = tf.reduce_sum(tf.complex(log_re, log_im), axis=1)
    return tf.exp(log)


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


class SequentialLSTMModel(SequentialDenseModel):

  @staticmethod
  def _create_model(d_in, d_out):
    add_dim = Lambda(lambda x: tf.expand_dims(x, axis=-1))
    return tf.keras.Sequential([Input(shape=(d_in,)), add_dim,
                                LSTM(10, return_sequences=True),
                                LSTM(d_out)])