import itertools
import numpy as np
import tensorflow as tf
from machines.autograd import base
from typing import Dict, Optional, Tuple


def initialize_rbm_params(w: Optional[np.ndarray] = None,
                          b: Optional[np.ndarray] = None,
                          c: Optional[np.ndarray] = None,
                          n_sites: Optional[int] = None,
                          n_hidden: Optional[int] = None,
                          std: float = 1e-2,
                          dtype = tf.float32) -> Dict[str, tf.Tensor]:
  params = {}
  if w is None:
    params["w_re"] = tf.Variable(
        np.random.normal(0.0, std, size=(n_sites, n_hidden)), dtype=dtype)
    params["w_im"] = tf.Variable(
        np.random.normal(0.0, std, size=(n_sites, n_hidden)), dtype=dtype)
  else:
    params["w_re"] = tf.Variable(w.real, dtype=dtype)
    params["w_im"] = tf.Variable(w.imag, dtype=dtype)

  if b is None:
    params["b_re"] = tf.Variable(np.zeros((n_hidden,)), dtype=dtype)
    params["b_im"] = tf.Variable(np.zeros((n_hidden,)), dtype=dtype)
  else:
    params["b_re"] = tf.Variable(b.real, dtype=dtype)
    params["b_im"] = tf.Variable(b.imag, dtype=dtype)

  if c is None:
    params["c_re"] = tf.Variable(np.zeros((n_sites,)), dtype=dtype)
    params["c_im"] = tf.Variable(np.zeros((n_sites,)), dtype=dtype)
  else:
    params["c_re"] = tf.Variable(c.real, dtype=dtype)
    params["c_im"] = tf.Variable(c.imag, dtype=dtype)

  return params


def logrbm_forward(v: tf.Tensor, w: tf.Tensor, b: tf.Tensor, c: tf.Tensor
                   ) -> tf.Tensor:
  """Forward prop of an RBM wavefunction.

  Args:
    v: Input units of shape (n_batch, n_sites,).
    w: Weights of shape (n_sites, n_hidden).
    b: Hidden biases of shape (n_hidden,).
    c: Visible biases of shape (n_sites,).

  Returns:
    Activations of shape (n_batch,)
  """
  cosh = tf.math.cosh(tf.matmul(v, w) + b[tf.newaxis])
  exp = tf.matmul(v, c[:, tf.newaxis])
  return exp[:, 0] + tf.reduce_sum(cosh, axis=-1)


def fit_rbm_to_dense(state: np.ndarray, n_hidden: int,
                     target_fidelity: float = 1e-4
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Fits an RBM to a dense state using gradient descent optimization.

  Args:
    state: Dense state array with shape (2**n_sites,).
    n_hidden: Number of hidden units in the RBM.
    target_fidelity: Required fidelity for the reconstruction.

  Returns:
    weights: Array of RBM weights with shape (n_hidden, n_sites).
    hidden biases: Array of RBM biases with shape (n_hidden,).
    visible biases: Array of RBM biases with shape (n_sites,).
  """
  n_sites = int(np.log2(len(state)))
  all_confs = np.array(list(itertools.product([-1, 1], repeat=n_sites)))
  all_confs = tf.convert_to_tensor(all_confs, dtype=tf.complex128)
  params = initialize_rbm_params(n_sites=n_sites, n_hidden=n_hidden,
                                 dtype=tf.float64)

  def get_vars():
    w = tf.complex(params["w_re"], params["w_im"])
    b = tf.complex(params["b_re"], params["b_im"])
    c = tf.complex(params["c_re"], params["c_im"])
    return w, b, c

  variables = list(params.values())
  optimizer = tf.keras.optimizers.Adam()
  target = tf.convert_to_tensor(state, dtype=tf.complex128)

  while True:
    with tf.GradientTape() as tape:
      w, b, c = get_vars()
      pred = tf.math.exp(logrbm_forward(all_confs, w, b, c))

      norm2 = tf.reduce_sum(tf.square(tf.abs(pred)))
      fidelity = tf.math.real(tf.reduce_sum(tf.math.conj(target) * pred))
      fidelity = fidelity / tf.math.sqrt(norm2)
      loss = 1 - fidelity

    grads = tape.gradient(loss, variables)
    if loss.numpy() < target_fidelity:
      break
    optimizer.apply_gradients(zip(grads, variables))

  print("RBM reconstruction fidelity:", fidelity.numpy())
  w, b, c = get_vars()
  return w.numpy(), b.numpy(), c.numpy()


class SmallRBMModel(base.BaseAutoGrad):

  def __init__(self, **kwargs):
    init_state = kwargs["init_state"]
    super(SmallRBMModel, self).__init__(**kwargs)
    self.name = "rbm_autograd"
    self.n_hidden = self.n_sites

    w, b, c = fit_rbm_to_dense(init_state, n_hidden=self.n_hidden,
                               target_fidelity=1e-4)

    self.w_re, self.w_im = self._add_variable(w)
    self.b_re, self.c_im = self._add_variable(b)
    self.c_re, self.c_im = self._add_variable(c)

  def _add_variable(self, v: np.ndarray) -> tf.Tensor:
    vt = np.array(self.time_steps * [v])
    re = self.add_variable(vt.real)
    im = self.add_variable(vt.imag)
    return re, im

  def complex_params(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    w = tf.complex(self.w_re, self.w_im)
    b = tf.complex(self.b_re, self.b_im)
    c = tf.complex(self.c_re, self.c_im)
    return w, b, c

  def forward(self):
    pass