import itertools
import numpy as np
import tensorflow as tf
from machines import base
from typing import Any, Dict


class CopiedRBM(base.Base):

  def __init__(self,
               initial_condition: np.ndarray,
               time_steps: int,
               n_hidden: int,
               dtype: Any = tf.float64,
               learning_rate: float = 1e-3):
    super(CopiedRBM, self).__init__(initial_condition, time_steps,
                                    dtype, learning_rate)
    self.n_sites = int(np.log2(len(self.initial_condition)))
    self.n_hidden = n_hidden
    self.init_params["n_hidden"] = self.n_hidden

    self.weights_keys = ["w", "b", "c"]
    self.weights_re = {k: tf.Variable(v, dtype=self.rtype)
                       for k, v in self._initialize_weights().items()}
    self.weights_im = {k: tf.Variable(v, dtype=self.rtype)
                       for k, v in self._initialize_weights().items()}
    self.variables = ([self.weights_re[k] for k in self.weights_keys] +
                      [self.weights_im[k] for k in self.weights_keys])

    all_confs = np.array(list(itertools.product([0, 1], repeat=self.n_sites)))
    # shape (n_sites, n_states)
    self.all_confs = tf.convert_to_tensor(all_confs.T, dtype=self.ctype)

    self.psi0 = tf.convert_to_tensor(initial_condition, dtype=self.ctype)
    self.psi0 = self.psi0[tf.newaxis] # (1, n_states)

  def _initialize_weights(self) -> Dict[str, np.ndarray]:
    w = np.random.normal(0, 1e-3, size=(self.time_steps, self.n_hidden,
                                        self.n_sites))
    b = np.zeros((self.time_steps, self.n_hidden))
    c = np.zeros((self.time_steps, self.n_sites))
    return {"w": w, "b": b, "c": c}

  def __call__(self) -> tf.Tensor:
    weights = {k: tf.complex(self.weights_re[k], self.weights_im[k])
               for k in self.weights_keys}
    # (time_steps, n_hidden, n_states)
    w_sigma = tf.einsum("thv,vi->thi", weights["w"], self.all_confs)
    #x = 2 * (w_sigma + weights["b"][:, :, tf.newaxis])
    #logcosh = tf.math.softplus(x) - x - np.log(2)
    cosh = tf.math.cosh(w_sigma + weights["b"][:, :, tf.newaxis])
    # (time_steps, n_states)
    c_sigma = tf.matmul(weights["c"], self.all_confs)
    # (time_steps, n_states)
    #psi = tf.exp(c_sigma + tf.reduce_sum(logcosh, axis=1))
    psi = tf.exp(c_sigma) * tf.reduce_prod(cosh, axis=1)
    return tf.concat([self.psi0, psi], axis=0)

  @classmethod
  def add_time_step(cls, old_machine: "CopiedRBM") -> "CopiedRBM":
    params = dict(old_machine.init_params)
    params["time_steps"] += 1
    new_machine = cls(**params)

    for k in old_machine.weights_keys:
      new = tf.concat([old_machine.weights_re[k],
                       old_machine.weights_re[k][-1][tf.newaxis]], axis=0)
      new_machine.weights_re[k].assign(new)

      new = tf.concat([old_machine.weights_im[k],
                       old_machine.weights_im[k][-1][tf.newaxis]], axis=0)
      new_machine.weights_im[k].assign(new)

    return new_machine
