import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from machines import base
from typing import Any, Dict


class CopiedFFNN(base.Base):

  def __init__(self, initial_condition: np.ndarray,
               time_steps: int,
               n_hidden: int,
               dtype: Any = tf.float32,
               learning_rate: float = 1e-3):
    super(CopiedFFNN, self).__init__(initial_condition, time_steps,
                                     dtype, learning_rate)
    self.n_hidden = n_hidden
    self.init_params["n_hidden"] = self.n_hidden
    self.n_sites = int(np.log2(len(initial_condition)))

    self.model_re = self._create_model()
    self.model_im = self._create_model()
    self.variables = self.model_re.variables + self.model_im.variables

    all_confs = np.array(list(itertools.product([0, 1], repeat=self.n_sites)))
    self.all_confs = tf.convert_to_tensor(all_confs, dtype=self.rtype)

    self.psi0 = tf.convert_to_tensor(initial_condition, dtype=self.ctype)
    self.psi0 = self.psi0[tf.newaxis] # (1, n_states)

  def _create_model(self) -> tf.keras.Model:
    model = tf.keras.models.Sequential()
    model.add(layers.Input((self.n_sites,)))
    n_hidden_total = self.time_steps * self.n_hidden
    model.add(layers.Dense(n_hidden_total, activation="relu"))
    model.add(layers.Reshape((n_hidden_total, 1)))
    model.add(layers.LocallyConnected1D(1, self.n_hidden,
                                        strides=self.n_hidden,
                                        activation="sigmoid"))
    return model

  def __call__(self) -> tf.Tensor:
    psi_re = self.model_re(self.all_confs)[:, :, 0]
    psi_im = self.model_im(self.all_confs)[:, :, 0]
    psi = tf.complex(psi_re, psi_im)
    return tf.concat([self.psi0, tf.transpose(psi)], axis=0)

  @staticmethod
  def get_weight_dict(model: tf.keras.Model) -> Dict[str, tf.Tensor]:
    d = {}
    d["dense_w"], d["dense_b"] = model.layers[0].variables
    d["lconn_w"], d["lconn_b"] = model.layers[2].variables
    return d

  @classmethod
  def add_time_step(cls, old_machine: "CopiedFFNN") -> "CopiedFFNN":
    params = dict(old_machine.init_params)
    params["time_steps"] += 1
    new_machine = cls(**params)

    # Assign weights of the new keras model from the old one
    # FIXME: Do the below weight assignment for `model_re` and `model_im`
    # self.model doesn't exist!
    old_weights = cls.get_weight_dict(old_machine.model)
    new_weights = cls.get_weight_dict(new_machine.model)

    new_dense_w = tf.concat([old_weights["dense_w"],
                             old_weights["dense_w"][:, -self.n_hidden:]],
                            axis=1)
    new_dense_b = tf.concat([old_weights["dense_b"],
                             old_weights["dense_b"][-self.n_hidden:]], axis=0)
    new_weights["dense_w"].assign(new_dense_w)
    new_weights["dense_b"].assign(new_dense_b)

    new_lconn_w = tf.concat([old_weights["lconn_w"],
                             old_weights["lconn_w"][-1][tf.newaxis]], axis=0)
    new_lconn_b = tf.concat([old_weights["lconn_b"],
                             old_weights["lconn_b"][-1][tf.newaxis]], axis=0)
    new_weights["lconn_w"].assign(new_lconn_w)
    new_weights["lconn_b"].assign(new_lconn_b)

    return new_machine
