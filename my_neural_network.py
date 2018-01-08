import numpy as np
import pandas as pd
import pickle
#import sklearn
from sklearn.datasets import load_iris
from activation_error import *
    
class NeuralNetwork:
    def __init__(self, x, y, weight_l, bias_l=None, activations=None):
        """Parameter
           ---------
           x            |  x in DataFrame format
           y            |  y in DataFrame format
           weight_l     |  the location of the weight(pickle), if it is a list of int, it is used to generate new weight
           bias_l       |  the location of the bias(pickle), if it is a list of int, it is used to generate new bias
           activations  |  a list of activation class instances
        """
        if type(weight_l) not in [list, str] or (type(bias_l) not in [list, str] and bias_l is not None):
            raise TypeError("Weight has to be list or str, bias has to be list, str or None")
        if type(weight_l) == type(bias_l) == list and weight_l != bias_l:
            raise ValueError("Length of weight and bias have to be the same, and elements have to be the same")
        if type(activations) != list:
            raise TypeError("Activations have to be a list of activation class instance")
        
        self.x, self.y = x, y
        self.read_w_b(weight_l, bias_l)
        
        if type(self.w) == list and len(activations) != len(self.w):
            raise ValueError("Length of activations have to be same as weight and bias")
        self.activations = activations
        
    
    def train(self, training_rate, n_iter, error_func=None, display_interval=None, batch=10, tol=1e-8,
              decreasing_rate=False, min_rate=1e-8):
        """error_func | the function for calculating error given y and predicted y, if None, mean square error is used
                      | must have a signature of (y, y_pred)
        """
        if error_func is None:
            error_func = MeanSquareError
        if not issubclass(error_func, ErrorFunc):
            return TypeError("error_func has to be subclass of ErrorFunc")
        if display_interval is None:
            display_interval = int(n_iter/100)
        
        errors, prev_error = {}, None
        Xs, Ys = np.array_split(np.asarray(self.x), batch), np.array_split(np.asarray(self.y), batch)
        for i in range(n_iter):
            error = sum([self.step(x, y, training_rate/batch, error_func) for x, y in zip(Xs, Ys)]) / batch
            if display_interval != 0 and i % display_interval == 0:
                errors[i] = error
            if (tol and prev_error and abs(prev_error - error) < tol) or training_rate < min_rate:
                break
            if decreasing_rate and prev_error and error > prev_error:
                training_rate *= 0.95
            
        return errors
                
    
    def predict(self, x):
        for w, b, a in zip(self.w, self.b, self.activations):
            x = a.act(x.dot(w) + b)
        return x
    
    def read_weight(self, weight_l):
        if type(weight_l) == str:
            with open(weight, "rb") as f:
                self.weight = pickle.load(f)
            self.weight_l = weight_l
        else: #list
            self.w = [2*np.random.rand(*shape)-1 for shape in zip([self.x.shape[1], *weight_l[:-1]], weight_l)]
            self.weight_l = "NeuralNetwork_wb.pkl"
    
    def read_bias(self, bias_l):
        if type(bias_l) == str:
            with open(bias_l, "rb") as f:
                self.bias = pickle.load(f)    
            self.bias_l = bias_l     #file location
        elif type(bias_l) == list:
            self.b = [2*np.random.rand(1, col)-1 for col in bias_l]
            self.bias_l = "NeuralNetwork_wb.pkl"
        else:  #None, follow size of weight
            self.b = [2*np.random.rand(1, weight.shape[1])-1 for weight in self.w]
    
    def read_w_b(self, weight_l, bias_l):
        if type(weight_l) == type(bias_l) == str and weight_l == bias_l:
            with open(weight_l, "rb") as f:
                self.w = pickle.load(f)
                self.b = pickle.load(f)
            self.weight_l = self.bias_l = weight_l
        else: # either they are from different location or should be generated from random
            self.read_weight(weight_l)
            self.read_bias(bias_l)
            
    def pickle_w_b(self, location=None):
        if location == None and self.weight_l == self.bias_l:
            with open(self.weight_l, "wb") as f:
                pickle.dump(self.w, f)
                pickle.dump(self.b, f)
        else:
            with open(location, "wb") as f:
                pickle.dump(self.w, f)
                pickle.dump(self.b, f)
            
    def pickle_network(self, location=None):
        if location == None:
             location = self.weight_l
        with open(location, "wb") as f:
            pickle.dump(self,f)
            
    def step(self, x, y, training_rate, error_func):
        layers = [x]
            
        for w_layer, b_layer, act_layer in zip(self.w, self.b, self.activations): #w, b and act of the layer
            layers.append(act_layer.act(layers[-1].dot(w_layer) + b_layer))
        
        for layer in range(len(self.w)-1, 0, -1):
            act_deriv = self.activations[layer].deriv(layers[layer+1])
            if len(act_deriv.shape) > 3:
                raise ValueError("act_deriv is too high dimension, dimension %s"%len(act_deriv.shape))
            if layer == len(self.w) - 1: # last layer
                if len(act_deriv.shape) < 3:
                    error_delta = error_func.deriv(self.y, layers[-1]) * act_deriv
                else: # == 3
                    error_delta = np.einsum('ij,ijk->ik', error_func.deriv(y, layers[-1]), act_deriv)
                
            else: # not last layer
                if len(act_deriv.shape) < 3:
                    error_delta = error_delta.dot(self.w[layer+1].T) * act_deriv
                else:
                    error_delta = np.einsum('ij,ijk->ik', error_delta.dot(self.w[layer+1]), act_deriv)
                
            self.w[layer] -= training_rate * layers[layer].T.dot( error_delta )
            self.b[layer] -= training_rate * np.sum(error_delta, axis=0)
        
        return error_func.error(y, layers[-1])
            

X, Y = load_iris(True)

np.random.seed(1048)
yy = [[1,0,0],[0,1,0],[0,0,1]]
my_Y = np.array([yy[i] for i in Y])
nn = NeuralNetwork(X, my_Y, [5, 6, 3], None, [ELU(1), ELU(1), MultiSoftmax])
nn.train(0.01, 3000, MeanRankedProbScore, 200, batch=2, decreasing_rate=True)

np.set_printoptions(suppress=True)
#print(my_Y)
print(nn.predict(X))