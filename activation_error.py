import numpy as np
import pandas as pd

class Activation:
    def __init__(self):
        pass
    
    @staticmethod
    def act(x):
        pass
    
    @staticmethod
    def deriv(y): # delta
        pass

class ErrorFunc:
    @staticmethod
    def error(y, predict_y):
        pass
    
    @staticmethod
    def deriv(y, predict_y):
        pass
    
class MeanSquareError(ErrorFunc):
    @staticmethod
    def error(y, predict_y):
        return ( (y - predict_y) ** 2 ).sum() / y.shape[0]
    
    @staticmethod
    def deriv(y, predict_y):
        return -2 / y.shape[0] * (y - predict_y)  # Nx1 array

class MeanRankedProbScore(ErrorFunc):
    @staticmethod
    def error(y, predict_y):
        return ( ( np.cumsum(y, axis=1) - np.cumsum(predict_y, axis=1) ) ** 2 / (y.shape[1]-1) ).sum() / y.shape[0]
    
    @staticmethod
    def deriv(y, predict_y):
        y_cumsum = np.cumsum(y, axis=1)
        predict_y_cumsum = np.cumsum(predict_y, axis=1)
        z = np.zeros_like(y, dtype=float)
        for i in range(y.shape[1]):
            z[:, i] = (y_cumsum[:, i:] - predict_y_cumsum[:, i:]).sum(axis=1)
        return -z / y.shape[0] / (y.shape[1] - 1)
    
class LogLoss(ErrorFunc):
    @staticmethod
    def error(y, predict_y):
        np.clip(predict_y, 1e-10, 1-1e-10, out=predict_y[:])
        return -np.sum(y * np.log(predict_y)) / y.shape[0]
    
    @staticmethod
    def deriv(y, predict_y):
        predict_y[predict_y > 1 - 1e-5] = 1 - 1e-5
        predict_y[predict_y < 1e-5] = 1e-5
        return (predict_y - y) / y.shape[0]
    
class Sigmoid(Activation):
    @staticmethod
    def act(x):
        return 1 / (1 + np.exp(x))
    
    @staticmethod
    def deriv(y):
        return y * (1 - y)
    
class TanH(Activation):
    @staticmethod
    def act(x):
        return 2 / (1 + np.exp(-2*x)) - 1
    
    @staticmethod
    def deriv(y):
        return 1 - y ** 2
    
class ReLU(Activation):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        
    def act(self, x):#, alpha=0.05
        z = x.copy()
        z[x < 0] = self.alpha * x[x < 0]
        return z
    
    def deriv(self, y): #, alpha=0.05
        z = np.ones_like(y, dtype=float)
        z[y < 0] = self.alpha
        return z

class ELU(Activation):
    def __init__(self, alpha=1):
        self.alpha = alpha
        
    def act(self, x):
        z = x.copy()
        z[x < 0] = self.alpha * (np.exp(x[x < 0]) - 1)
        return z
    
    def deriv(self, y):
        z = np.ones_like(y, dtype=float)
        z[y < 0] = y[y < 0] + self.alpha
        return z

class SinAct(Activation):
    @staticmethod
    def act(x):
        return np.sin(x)

    @staticmethod
    def deriv(y):
        return np.cos(x)

class MultiSoftmax(Activation):
    @staticmethod
    def act(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(x.shape[0],1)
    
    @staticmethod
    def deriv(y):
        z = - np.einsum('ij,ik->ijk', y, y) #simplified
        z[:, np.arange(y.shape[1]), np.arange(y.shape[1])] += y
        return z
    
    @staticmethod
    def delta(current_x_layer, previous_x_layer, error_delta):
        return previous_x_layer.T.dot( error_delta * act_layer.deriv(current_x_layer) )