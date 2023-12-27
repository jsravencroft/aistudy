class Layer:
    def __init__(self):
      self.input = None   # Prior Layer
      self.output = None  # Next Layer

    # Will ouptut a Y for a given X value
    def forward_propagation(self, input):
      raise NotImplementedError
  
    # Computes dE/dX for a given dE/dY and update parameters if present
    def backward_propagation(self, output_error, learning_rate):
      raise NotImplementedError
