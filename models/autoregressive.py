"""Autoregressive models with built-in norm conservation."""

import numpy as np
import itertools
import tensorflow as tf
from tensorflow.keras.layers import Activation, Input, Reshape, Lambda
from models import base


class MaskedAffineLayer(tf.keras.layers.Layer):
  """Affine layer for a masked regular feed-forward NN.

  Each component i in output is connected only to earliest (j < i) components
  of the input.

  AUTO-BACKPROP DOESN'T WORK (GIVES NONE GRADIENT)
  """

  def __init__(self, num_outputs, num_inputs):
    super(MaskedAffineLayer, self).__init__()
    self.num_outputs = num_outputs

    if self.num_outputs % num_inputs != 0:
      raise NotImplementedError("Number of outputs must be a multiple of the "
                                "number of inputs.")
    upscaling_factor = self.num_outputs // num_inputs

    # Create mask as if upscaling_factor == 1
    weight_mask = np.tril(np.ones([num_inputs, num_inputs]))
    # Repeat the rows accordingly
    weight_mask = np.repeat(weight_mask, repeats=upscaling_factor, axis=0)

    full_weights = self.add_variable("full_weights",
                                     shape=[num_inputs, self.num_outputs])
    # Cast mask to tensorflow add transpose to make dims agree with weights
    weight_mask = tf.cast(weight_mask.T, dtype=full_weights.dtype)

    # The effective weights are just the "lowest triangular" part of the
    # full weight matrix
    self.effective_weights = weight_mask * full_weights

  def build(self, input_shape):
    if len(input_shape) > 2:
      raise NotImplementedError("Masked affine layer is implemented only for "
                                "one-dimensional data.")

    if int(self.effective_weights.shape[0]) != input_shape[-1]:
      raise ValueError("Input has incompatible shape.")

  def call(self, input):
    return tf.matmul(input, self.effective_weights)


class FullAutoregressiveModel(base.BaseModel):
  """Independent weights used for each time step."""

  def __init__(self, init_state, time_steps, std=1e-3,
               rtype=tf.float32, ctype=tf.complex64):
    n_sites = int(np.log2(len(init_state)))
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)

    all_confs = np.array(list(itertools.product([0, 1], repeat=n_sites)))
    self.all_confs_oh = self._convert_to_onehot(all_confs, rtype)
    inputs = np.concatenate([np.ones((len(all_confs), 1)),
                             all_confs[:, :-1]], axis=1)
    self.inputs = tf.cast(inputs, dtype=rtype)

    w_shapes = [(n_sites, 20), (20, 60), (60, n_sites * time_steps * 2)]

    self.w_norm = [tf.Variable(np.random.normal(0, std, size=s), dtype=rtype)
                   for s in w_shapes]
    self.w_phase = [tf.Variable(np.random.normal(0, std, size=s), dtype=rtype)
                    for s in w_shapes]
    self.masks = [tf.cast(np.triu(np.ones(s)), rtype) for s in w_shapes]

    self.vars = [self.w_norm, self.w_phase]

  @staticmethod
  def _convert_to_onehot(all_confs, dtype=tf.float32):
    shape = all_confs.shape
    n = np.prod(shape)
    one_hot = np.zeros((n, 2))
    one_hot[np.arange(n), all_confs.ravel()] = 1
    return tf.cast(one_hot.reshape(shape + (2,)), dtype=dtype)

  @staticmethod
  def masked_affine(w, x, mask):
    return tf.matmul(x, mask * w)

  def variational_wavefunction(self, training=False):
    norms = tf.cast(self.inputs, dtype=self.inputs.dtype)
    phases = tf.cast(self.inputs, dtype=self.inputs.dtype)
    for i in range(len(self.masks) - 1):
      norms = tf.nn.relu(
          self.masked_affine(self.w_norm[i], norms, self.masks[i]))
      phases = tf.nn.relu(
          self.masked_affine(self.w_phase[i], norms, self.masks[i]))
    norms = tf.sigmoid(
        self.masked_affine(self.w_norm[-1], norms, self.masks[-1]))
    phases = tf.sigmoid(
        self.masked_affine(self.w_phase[-1], norms, self.masks[-1]))

    output_shape = (len(self.inputs), self.time_steps, self.n_sites, 2)
    norms = tf.reshape(norms, output_shape)
    phases = tf.reshape(phases, output_shape)

    norms_i = np.einsum("btis,bis->tbi", norms, self.all_confs_oh)
    normalization_i = tf.reduce_sum(tf.exp(2 * norms), axis=-1)
    phases = 2 * np.pi * np.einsum("btis,bis->tb", phases, self.all_confs_oh)

    logpsi_re = tf.reduce_sum(norms_i - 0.5 * tf.log(normalization_i), axis=-1)
    return tf.exp(tf.complex(logpsi_re, phases))