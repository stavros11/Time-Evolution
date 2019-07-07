"""Base model that can be optimized using tf_optimization.py"""
import tensorflow as tf


class BaseModel:
  """Base Clock state model. All TF models should inherit this.

  Methods that need to be implemented when defining a new model:
    __init__ : Make sure to define self.init_state as a tf tensor with an
      additional axis here. It is required for full wavefunction calculation.
    variational_wavefunction
    Optionally: update (if we want to use hard-coded grads instead of autodiff)
  """

  def __init__(self, init_state, time_steps,
               rtype=tf.float32, ctype=tf.complex64):
    """Initialized model according to given parameters.

    Args:
      init_state: Initial state as a numpy array of shape (2**N,).
      time_steps: Number of time steps for time discretization.
      rtype: TensorFlow real float type.
      ctpye: TensorFlow complex float type.
    """
    # The attribute self.init_state must be defined here!
    # The attribute self.vars must be defined here!
    raise NotImplementedError

  def variational_wavefunction(self, training=False):
    """Calculates the current full wavefunction of the model.

    This excludes the initial condition which is kept constant during
    optimization.

    Args:
      training: If True the graph is watched for backprop
        (not relevent to all models).

    Returns:
      Full wavefunction with shape (M, 2**N) as a tensorflow tensor.
    """
    raise NotImplementedError

  def update(self, optimizer, updater):
    """Single update step of the model variables using autodiff.

    Depending on the model, we may want to replace this method with an
    update using hard-coded gradients. In that case the updater can be a
    function that gives the complex gradients.

    Args:
      optimizer: TensorFlow optimizer object to perform the update.
      updater: Function that calculates Eloc loss for the Clock Hamiltonian.
    """
    with tf.GradientTape() as tape:
      full_psi = self.wavefunction(training=True)
      eloc = updater(full_psi)

    grads = tape.gradient(eloc, self.vars)
    optimizer.apply_gradients(zip(grads, self.vars))
    return eloc

  def wavefunction(self, training=False):
    """Calculates the current full wavefunction of the model.

    Args:
      training: If True the graph is watched for backprop
        (not relevent to all models).

    Returns:
      Full wavefunction with shape (M + 1, 2**N) as a tensorflow tensor.
    """
    state_t = self.variational_wavefunction(training=training)
    return tf.concat([self.init_state, state_t], axis=0)

  def overlap(self, state, normalize_states=False):
    """Calculates overlap between current state and a given state.

    Args:
      state: Given state to calculate overlap with as a tf tensor.
        Must be of the same complex type as the model's state.
      normalize_states: Normalize states before calculating the overlap.

    Returns:
      Overlap as a tf scalar.
    """
    var_state = self.wavefunction()
    prod = tf.reduce_sum(tf.conj(state) * var_state)
    if normalize_states:
      norm1 = tf.reduce_sum(tf.square(tf.abs(state)))
      norm2 = tf.reduce_sum(tf.square(tf.abs(var_state)))
      return tf.square(tf.abs(prod)) / (norm1 * norm2)
    return tf.square(tf.abs(prod))