import numpy as np
from activations import Activations

class Propagation: 

    def forward(self, x, activations, drop_ratio):

        self.Z, self.A, self.drops = {}, {}, {}
        self.A[0] = x.copy()

        for i in range(self.ns-1):

            keep_ratio = 1-drop_ratio[i] 
            activation = getattr(Activations, activations[i]) 
            self.Z[i+1] = np.matmul(self.A[i], self.W[i+1]) + self.B[i+1]
            self.A[i+1] = activation(self, self.Z[i+1])
            d = np.random.rand(self.size[i+1]) < keep_ratio 
            self.drops[i+1] = d 
            self.A[i+1] *= d 
            self.A[i+1] /= keep_ratio 

        activation = getattr(Activations, activations[self.ns-1])  
        self.Z[self.ns] = np.matmul(self.A[self.ns-1], self.W[self.ns]) + self.B[self.ns]   
        self.A[self.ns] = activation(self, self.Z[self.ns])
        return self.A[self.ns] 
    
    
    def backward(self, x, y, activations, drop_ratio):

        self.dW, self.dB, self.dZ, self.dA = {}, {}, {}, {}
        L = self.ns 
        self.dZ[L] = self.A[L] - y 

        for i in range(L, 0, -1): 

            self.dW[i] = np.matmul(self.A[i-1].T, self.dZ[i])
            self.dB[i] = np.sum(self.dZ[i],axis = 0, keepdims = True)

            if i != 1:
                keep_ratio = 1 - drop_ratio[i-2] 
                activation = getattr(Activations, activations[i-2])   
                self.dA[i-1] = np.matmul(self.dZ[i], self.W[i].T)
                self.dA[i-1] *= self.drops[i-1] 
                self.dA[i-1] /= keep_ratio 
                self.dZ[i-1] = np.multiply(self.dA[i-1], activation(self, self.A[i-1], grad = True))
                