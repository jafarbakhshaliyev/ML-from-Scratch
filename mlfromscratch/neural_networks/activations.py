import numpy as np

class Activations:
    
    def sigmoid(self, z, grad = False):
        if grad == True: return z*(1-z) 
        return 1/(1+np.power(np.e, -z))
    
    def tanh(self, z, grad = False):
        if grad == True: return 1.0 - z**2 
        return np.tanh(z)
    
    def relu(self, z, grad = False):
        if grad == True: return 1.0 * (z > 0)
        return z * (z > 0)
    
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True) 
