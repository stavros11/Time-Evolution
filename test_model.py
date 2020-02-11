import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

n_sites = 6
time_steps = 21
n_hidden = 6


#model = tf.keras.models.Sequential()

#model.add(layers.Input((n_sites,)))
#model.add(layers.Dense(time_steps * n_hidden, activation="relu"))
#model.add(layers.Reshape((time_steps * n_hidden, 1)))
#model.add(layers.Dense(1, activation="sigmoid"))
#model.add(layers.LocallyConnected1D(1, n_hidden, strides=n_hidden))

#print(model.summary())


import machines


init_state = np.random.random([21, 2**n_sites])


model = machines.CopiedFFNN(init_state, n_hidden=n_hidden)
print(model)
print(model.dense)
