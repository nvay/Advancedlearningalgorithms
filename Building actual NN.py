import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('./deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)        #(price in 1000s of dollars)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, c='r', marker='x', label='Data Point')
ax.legend( fontsize='x-large', loc='upper left')
ax.set_xlabel('(size in 1000 square feet)', fontsize='xx-large')
ax.set_ylabel('(size in 1000 square feet)', fontsize=15)
ax.set_ylabel('(price in 1000s of dollars)')
plt.show()

linear_layer = tf.keras.layers.Dense(1, activation= 'linear')

linear_layer.get_weights()

al = linear_layer(X_train[0].reshape(1, 1))
print(al)