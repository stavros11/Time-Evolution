import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from machines.autograd import base


class SpinTime:

  def __init__(self, machine: base.BaseAutoGrad,
               n_samples: int, n_corr: int, n_burn: int):
    self.n_samples = int(n_samples)
    self.n_corr = int(n_corr)
    self.n_burn = int(n_burn)

    n_sites = machine.n_sites
    time_steps = machine.time_steps + 1

    init_state = np.random.randint(0, 2, n_sites + 1)
    init_state[:-1] = 2 * init_state[-1] - 1
    init_state[-1] = np.random.randint(0, time_steps)
    self.init_state = tf.convert_to_tensor(init_state, dtype=tf.int32)

    def log_prob(x: tf.Tensor) -> tf.Tensor:
      psi = machine.forward_log(x[tf.newaxis, :-1], x[tf.newaxis, -1])
      return tf.math.real(psi[0])

    def new_state(x: tf.Tensor, seed: int) -> tf.Tensor:
      old_state = x[0]
      flipper = np.ones(n_sites)
      flipper[np.random.randint(n_sites)] = -1

      new_spins = old_state[:-1] * flipper
      new_time = old_state[-1] + np.random.randint(0, time_steps)
      new_time = tf.math.floormod(new_time, time_steps)
      return [tf.concat([new_spins, new_time[tf.newaxis]], axis=0)]

    # Initialize the HMC transition kernel.
    self.kernel = tfp.mcmc.RandomWalkMetropolis(target_log_prob_fn=log_prob,
                                                new_state_fn=new_state)

  def __call__(self):
    samples = tfp.mcmc.sample_chain(
        num_results=self.n_samples,
        num_burnin_steps=self.n_burn,
        current_state=self.init_state,
        kernel=self.kernel,
        trace_fn=None,
        parallel_iterations=4)

    self.init_state = samples[-1]
    return samples