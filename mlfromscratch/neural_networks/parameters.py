import numpy as np

class Parameters:
    
    def initialize_parameters(self, X, y, neuron_sizes):
        self.nx = X.shape[1]
        self.ny = y.shape[1]
        self.ns = len(neuron_sizes)
        self.size = [self.nx] + neuron_sizes
        self.W, self.B = {}, {}
        for i in range(self.ns):
            self.W[i+1] = np.random.randn(self.size[i], self.size[i+1]) * math.sqrt(2/self.size[i]) # He initialization
            self.B[i+1] = np.zeros((1, self.size[i+1])) 
        return self
    
    def update_parameters(self, X, dW, dB, learning_rate): 
        for i in range(self.ns):
            self.W[i+1] -= learning_rate*(dW[i+1]/self.nx)
            self.B[i+1] -= learning_rate*(dB[i+1]/self.nx)
        return self 