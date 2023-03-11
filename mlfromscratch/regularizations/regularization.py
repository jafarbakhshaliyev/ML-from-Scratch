import numpy as np

class Regularization:
    def __init__(self) -> None:
     pass

    def L2(self, theta, lambd):
      return lambd*theta.T@theta, 2*lambd*theta, 2*lambd
    
    def L1(self, theta, lambd):
      return lambd*np.sum(np.abs(theta)), lambd*np.sign(theta), 0
    
    def ElasticNet(self, theta, lambd, ratio):
      return ratio*lambd*theta.T@theta + (1-ratio)*lambd*np.sum(np.abs(theta)), ratio*2*lambd*theta + (1-ratio)*lambd*np.sign(theta), ratio*2*lambd
    
 