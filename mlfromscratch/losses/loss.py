import numpy as np

class Loss:
    def __init__(self) -> None:
       pass
    
    def mse(self, X, y, theta):
       return ((X@theta-y)**2).mean(), 1.0/len(y)*X.T.dot(X.dot(theta) - y), 1.0/len(y)*X.T@X
    
    def binary_cross_entropy(self, X, y, theta):
       
       epsilon = 1e-9
       p = 1/(1+np.exp(-X@theta))
       p = np.clip(p, epsilon, 1. - epsilon) 
       W = np.diag((p*(1-p)).reshape(-1))
       
       return -1.0/len(y)*(y.T@np.log(p) + (1-y).T@np.log((1-p))),-1.0/len(y)*X.T.dot(y - p), 1.0/len(y)*X.T@W@X

 