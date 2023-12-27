class Network:

    def __init__(self):
      self.layers = []
      self.loss   = None
      self.loss_prime = None


    def add_layer(self, layer):
      self.layers.append(layer)

    def set_loss_algorithm(self, loss, loss_prime):
      self.loss       = loss
      self.loss_prime = loss_prime

    def train(self, x_train, y_train, epochs, learning_rate):
      samples = len(x_train)

      # TODO = I'm not sure range is required here
      for i in range(epochs):
        err = 0

        for j in range(samples):
          output = x_train[j]
        
          # Forward
          for layer in self.layers:
            output = layer.forward_propagation(output)
          
          err += self.loss(y_train[j], output)
           
          # Backwards
          error = self.loss_prime(y_train[j], output)
          for layer in reversed(self.layers):
            error = layer.backward_propagation(error, learning_rate)

        err = err / samples
        print('Epoch %d/%d error =%f' % (i + 1, epochs, err))
     
    def predict(self, input_data):
      samples = len(input_data)
      result  = [] 

      for i in range(samples):
        output = input_data[i]
        for layer in self.layers:
          output = layer.forward_propagation(output)

        result.append(output)
      return result
