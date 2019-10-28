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
  logcosh = tf.math.log(tf.math.cosh(tf.matmul(v, w) + b[tf.newaxis]))
  exp = tf.matmul(v, c[:, tf.newaxis])
  return exp[:, 0] + tf.reduce_sum(logcosh, axis=-1)


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

    self._create_variables(init_state)
    all_confs = np.array(list(itertools.product([-1, 1], repeat=self.n_sites)))
    self.all_confs = tf.convert_to_tensor(all_confs, dtype=self.output_type)

  def _create_variables(self, init_state: np.ndarray):
    w, b, c = fit_rbm_to_dense(init_state, n_hidden=self.n_hidden,
                               target_fidelity=1e-4)
    self._add_variables(np.array(self.time_steps * [w]),
                        np.array(self.time_steps * [b]),
                        np.array(self.time_steps * [c]))

  def _add_variables(self, wt: np.ndarray, bt: np.ndarray, ct: np.ndarray
                     ) -> tf.Tensor:
    add_variable = lambda v: (self.add_variable(v.real),
                              self.add_variable(v.imag))
    self.w_re, self.w_im = add_variable(wt)
    self.b_re, self.b_im = add_variable(bt)
    self.c_re, self.c_im = add_variable(ct)

  def complex_params(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    w = tf.complex(self.w_re, self.w_im)
    b = tf.complex(self.b_re, self.b_im)
    c = tf.complex(self.c_re, self.c_im)
    return w, b, c

  def logforward(self):
    w, b, c = self.complex_params()
    csigma = tf.einsum("tj,bj->tb", c, self.all_confs)
    wsigma = tf.einsum("tji,bj->tbi", w, self.all_confs)
    logcosh = tf.math.log(tf.math.cosh(wsigma + b[:, tf.newaxis]))
    return csigma + tf.reduce_sum(logcosh, axis=-1)

  def forward(self):
    psi = tf.math.exp(self.logforward())
    return tf.concat([self.init_state, psi], axis=0)

  @staticmethod
  def _add_time_step_to_variable(v: np.ndarray) -> np.ndarray:
    new_shape = (len(v) + 1,) + v.shape[1:]
    new_v = np.zeros(new_shape, dtype=v.dtype)
    new_v[:-1] = np.copy(v)
    new_v[-1] = np.copy(v[-1])
    return new_v

  def add_time_step(self):
    self.variables = []
    self.time_steps += 1

    old_params = self.complex_params()
    arg_names = ["wt", "bt", "ct"]
    old_params = {k: self._add_time_step_to_variable(x.numpy())
                  for k, x in zip(arg_names, old_params)}
    self._add_variables(**old_params)


class SmallRBMProductPropModel(SmallRBMModel):

  def _create_variables(self, init_state: np.ndarray):
    w, b, c = fit_rbm_to_dense(init_state, n_hidden=self.n_hidden,
                               target_fidelity=1e-4)
    self.w0 = tf.convert_to_tensor(w.ravel()[:, np.newaxis],
                                   dtype=self.output_type)
    self.b0 = tf.convert_to_tensor(b[:, np.newaxis], dtype=self.output_type)
    self.c0 = tf.convert_to_tensor(c[:, np.newaxis], dtype=self.output_type)

    self.propagators = {
        "w_re": self.add_variable(self._initialize(
            self.n_hidden * self.n_sites, True)),
        "w_im": self.add_variable(self._initialize(
            self.n_hidden * self.n_sites, False)),
        "b_re": self.add_variable(self._initialize(self.n_hidden, True)),
        "b_im": self.add_variable(self._initialize(self.n_hidden, False)),
        "c_re": self.add_variable(self._initialize(self.n_sites, True)),
        "c_im": self.add_variable(self._initialize(self.n_sites, False))}

  @staticmethod
  def _initialize(n: int, close_to_eye: bool = False,
                  std: float = 1e-2) -> np.ndarray:
    ident = np.eye(n)
    noise = np.random.normal(0.0, std, size=ident.shape)
    if close_to_eye:
      return ident + noise
    return noise

  def complex_params(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    params = {"w": [self.w0], "b": [self.b0], "c": [self.c0]}
    props = {k: tf.complex(self.propagators["{}_re".format(k)],
                           self.propagators["{}_im".format(k)])
             for k in params.keys()}

    for _ in range(self.time_steps):
      for k in params.keys():
        params[k].append(tf.matmul(props[k], params[k][-1]))

    for k in params.keys():
      params[k] = tf.stack(params[k][1:])[:, :, 0]

    w_shape = (self.time_steps, self.n_sites, self.n_hidden)
    w = tf.reshape(params["w"], w_shape)
    return w, params["b"], params["c"]

  def add_time_step(self):
    self.time_steps += 1