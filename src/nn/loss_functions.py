import numpy as np

def mse(y_true, y_prediction):
  return np.mean(np.power(y_true - y_prediction, 2));


def mse_prime(y_true, y_prediction):
  return 2*(y_prediction - y_true)/y_true.size
