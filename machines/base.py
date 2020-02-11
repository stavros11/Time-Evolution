import numpy as np
import tensorflow as tf
from typing import Any, Callable


class Base:

  def __init__(self, initial_condition: np.ndarray,
               time_steps: int,
               dtype: Any = tf.float64,
               learning_rate: float = 1e-3):
    self.rtype = dtype
    self.time_steps = time_steps
    self.initial_condition = initial_condition
    self.init_params = {"initial_condition": initial_condition,
                        "time_steps": time_steps,
                        "dtype": dtype,
                        "learning_rate": learning_rate}

    self._dense = None
    self.mask = None

    self.optimizer = tf.keras.optimizers.Adam(learning_rate)

  def __call__(self) -> tf.Tensor:
    raise NotImplementedError

  @classmethod
  def add_time_step(cls, old_machine: "Base") -> "Base":
    raise NotImplementedError

  @property
  def ctype(self) -> Any:
    if self.rtype == tf.float32:
      return tf.complex64
    elif self.rtype == tf.float64:
      return tf.complex128
    else:
      raise TypeError

  @property
  def dense(self) -> tf.Tensor:
    if self._dense is None:
      self._dense = self.__call__()
    return self._dense

  def update(self, objective: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    with tf.GradientTape() as tape:
      self._dense = self.__call__()
      loss = objective(self._dense)

    grads = tape.gradient(loss, self.variables)
    if self.mask is not None:
      grads = [self.mask * g for g in grads]
    self.optimizer.apply_gradients(zip(grads, self.variables))
    return loss


class FullWavefunction(Base):

  def __init__(self, initial_condition: np.ndarray,
               time_steps: int,
               dtype: Any = tf.float64,
               learning_rate: float = 1e-3):
    super(FullWavefunction, self).__init__(initial_condition,
                                           time_steps,
                                           dtype,
                                           learning_rate)
    initial_state = np.array((self.time_steps + 1) * [initial_condition])
    self.var_re = tf.Variable(initial_state.real, dtype=self.rtype)
    self.var_im = tf.Variable(initial_state.imag, dtype=self.rtype)
    self.variables = [self.var_re, self.var_im]

    mask = np.ones_like(initial_state)
    mask[0] = 0.0
    self.mask = tf.convert_to_tensor(mask, dtype=dtype)

  def __call__(self) -> tf.Tensor:
    return tf.complex(self.var_re, self.var_im)

  @classmethod
  def add_time_step(cls, old_machine: "FullWavefunction") -> "FullWavefunction":
    params = dict(old_machine.init_params)
    params["time_steps"] += 1
    new_machine = cls(**params)

    new_var_re = tf.concat([old_machine.var_re,
                            old_machine.var_re[-1][tf.newaxis]], axis=0)
    new_machine.var_re.assign(new_var_re)
    new_var_im = tf.concat([old_machine.var_im,
                            old_machine.var_im[-1][tf.newaxis]], axis=0)
    new_machine.var_im.assign(new_var_im)

    return new_machine


class FullProp(Base):

  def __init__(self,
               initial_condition: np.ndarray,
               time_steps: int,
               dtype: Any = tf.float64,
               learning_rate: float = 1e-3):
    super(FullProp, self).__init__(initial_condition, time_steps,
                                   dtype, learning_rate)
    n_states = len(self.initial_condition)

    self.u_re = tf.Variable(np.eye(n_states), dtype=self.rtype)
    self.u_im = tf.Variable(np.eye(n_states), dtype=self.rtype)
    self.variables = [self.u_re, self.u_im]

    self.psi0 = tf.convert_to_tensor(initial_condition, dtype=self.ctype)

  def __call__(self) -> tf.Tensor:
    states = [self.psi0[:, tf.newaxis]]
    u = tf.complex(self.u_re, self.u_im)
    for _ in range(self.time_steps):
      states.append(tf.matmul(u, states[-1]))
    return tf.stack(states)[:, :, 0]

  @classmethod
  def add_time_step(cls, old_machine: "FullProp") -> "FullProp":
    old_machine.time_steps += 1
    return old_machine
