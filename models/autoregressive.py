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
  
  def __init__(self, init_state, time_steps, 
               rtype=tf.float32, ctype=tf.complex64):
    n_sites = int(np.log2(len(init_state)))
    self.init_state = tf.cast(init_state[np.newaxis], dtype=ctype)
        
    
    all_confs = np.array(list(itertools.product([0, 1], repeat=n_sites)))
    self.all_confs_oh = self._convert_to_onehot(all_confs[np.newaxis], rtype)
    
    inputs = np.concatenate(
        [np.ones([len(all_confs), 1], dtype=all_confs.dtype), 
         all_confs[:, 1:]], axis=1)
    self.inputs = tf.cast(inputs, dtype=rtype)
    
    self.norm_nn = self._create_model(n_sites, time_steps)
    self.phase_nn = self._create_model(n_sites, time_steps)
    
    self.vars = (self.norm_nn.trainable_variables + 
                 self.phase_nn.trainable_variables)

  @staticmethod
  def _convert_to_onehot(all_confs, dtype=tf.float32):
    shape = all_confs.shape
    n = np.prod(shape)
    one_hot = np.zeros((n, 2))
    one_hot[np.arange(n), all_confs.ravel()] = 1
    return tf.cast(one_hot.reshape(shape + (2,)), dtype=dtype)

  @staticmethod
  def _create_model(n_sites, time_steps, d_spin=2):
    n_hidden = n_sites * time_steps
    transpose_layer = Lambda(lambda x: tf.transpose(x, [2, 0, 1, 3]))
    return tf.keras.Sequential([Input((n_sites,)),
                                MaskedAffineLayer(n_hidden, n_sites),
                                Activation("relu"),
                                MaskedAffineLayer(n_hidden * d_spin, n_hidden),
                                Activation("sigmoid"),
                                Reshape((n_sites, time_steps, 2)),
                                transpose_layer])

  def variational_wavefunction(self, training=False):
    norms = self.norm_nn(self.inputs, training=training)
    phases = self.phase_nn(self.inputs, training=training)
    
    norms_i = tf.reduce_sum(norms * self.all_confs_oh, axis=-1)
    normalization_i = tf.reduce_sum(tf.square(norms), axis=-1)
    
    logpsi_re = tf.reduce_sum(tf.log(norms_i) - 0.5 * tf.log(normalization_i), 
                              axis=-1)
    logpsi_im = 2 * np.pi * tf.reduce_sum(phases * self.all_confs_oh, 
                                          axis=(-2, -1))
    return tf.exp(tf.complex(logpsi_re, logpsi_im))