import numpy as np



# TODO Bundle these into map of entities by name with f and f'
def tanh(x):
  return np.tanh(x)

def tanh_prime(x):
   return 1 - np.tanh(x)**2


class Tanh:

  @staticmethod
  def forward(x):
    return np.tanh(x)

  @staticmethod
  def backward(x):
   return 1 - np.tanh(x)**2
  
