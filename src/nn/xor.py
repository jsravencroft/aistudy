# Super simple test run
import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer

import activation_functions
import loss_functions

# Build our network
net = Network()
net.add_layer(FCLayer(2, 3))
net.add_layer(ActivationLayer(activation_functions.Tanh))
net.add_layer(FCLayer(3, 1))
net.add_layer(ActivationLayer(activation_functions.Tanh))

# Training data
training_data    = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
training_answers = np.array([[[0]], [[1]], [[1]], [[0]]])

# Train the network
net.set_loss_algorithm(loss_functions.mse, loss_functions.mse_prime)
net.train(training_data, training_answers, epochs=5000, learning_rate=0.01)

# Test our network
out = net.predict(training_data)
print(out)
