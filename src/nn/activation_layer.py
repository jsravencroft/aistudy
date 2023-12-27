# This could be baked into FC Layers, but separate is nice, too.
from layer import Layer

class ActivationLayer(Layer):
  # Activation for forward
  # Activation' for backwards
  #def __init__(self, activation, activation_prime):
  def __init__(self, activation):
    self.activation_forward  = activation.forward
    self.activation_backward = activation.backward

  def forward_propagation(self, input_data):
    # We must save this for later during back propagation
    self.input = input_data
    # TODO = Verify it's OK with just input_data
    self.output = self.activation_forward(self.input)

    return self.output

  def backward_propagation(self, output_error, learning_rate):
    return self.activation_backward(self.input) * output_error
