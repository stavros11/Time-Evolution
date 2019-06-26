import numpy as np
import tensorflow as tf
import utils
from models import autoregressive
tf.enable_v2_behavior()

# System parameters
n_sites = 4
time_steps = 100
t_final = 1.0
h_init = 0.5
h_ev = 1.0

# Optimization parameters
ctype = tf.complex64
n_epochs = 12000
n_message = 20
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)


# Find exact evolution state
t_grid = np.linspace(0.0, t_final, time_steps + 1)
dt = t_grid[1] - t_grid[0]
exact_state, obs = utils.tfim_exact_evolution(n_sites, t_final, time_steps,
                                              h0=h_init, h=h_ev)
exact_state_tf = tf.cast(exact_state, dtype=ctype)


# Define model
model = autoregressive.FullAutoregressiveModel(exact_state[0], time_steps)
test_wvf = model.wavefunction()
wvf = test_wvf.numpy()

print(wvf.shape)
print((np.abs(wvf)**2).sum(axis=1))