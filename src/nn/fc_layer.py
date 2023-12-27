import logging
from layer import Layer
import numpy as np

# TODO - Environment variable or maybe per-payer observability.
logging.basicConfig(encoding='utf-8', level=logging.WARN)

class FCLayer(Layer):

  def __init__(self, input_size, output_size):
    # TODO - Capture these values per layer at the end of training
    # These random values are random
    self.weights = np.random.rand(input_size, output_size) - 0.5
    self.bias    = np.random.rand(1, output_size) - 0.5
    logging.debug('Initial weights are: %s' % self.weights)
    logging.debug('Initial bias is: %s' % self.bias)

  
  # TODO - Tests
  def forward_propagation(self, input_data):
    self.input  = input_data
    self.output = np.dot(self.input, self.weights) + self.bias
    logging.debug('Output is: %s' % self.output)
    return self.output


  # Computes dE/dW, dE/dB for given output_error=dE/dY
  # Comes from downstream layer during error correction
  def backward_propagation(self, output_error, learning_rate):
    input_error   = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)

    # Update weights & biases
    self.weights -= learning_rate * weights_error
    self.bias    -= learning_rate * output_errors

    logging.debug('Updated weights are: %s' % self.weights)
    logging.debug('Updated bias is: %s' % self.bias)

  
  # TODO - Tests
  def forward_propagation(self, input_data):
    self.input  = input_data
    self.output = np.dot(self.input, self.weights) + self.bias
    logging.debug('Output is: %s' % self.output)
    return self.output


  # Computes dE/dW, dE/dB for given output_error=dE/dY
  # Comes from downstream layer during error correction
  def backward_propagation(self, output_error, learning_rate):
    input_error   = np.dot(output_error, self.weights.T)
    weights_error = np.dot(self.input.T, output_error)

    # Update weights & biases
    self.weights -= learning_rate * weights_error
    self.bias    -= learning_rate * output_error

    # Returns the error on the input to pass upstream, who will then do their own backward_propagation
    return input_error 
