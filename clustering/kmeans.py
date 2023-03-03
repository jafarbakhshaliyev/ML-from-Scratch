import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    
    def fit(self, X, k, maxiter = 500, seed = 124, epsilon = 10e-5):

        np.random.seed(seed)
        self.k = k
        self.mu = np.zeros((k, X.shape[1]))
        self.mu[0,:] = X[np.random.randint(X.shape[0])].reshape(1,-1)

        for i in range(1, k):
            self.mu[i,:] = X[np.sum([np.sum((X - self.mu[i,:])**2, axis = 1) for i in range(0,(i-1))], axis = 0).argmax()].reshape(1,-1)
    
        for iter_ in range(maxiter): 

            mu_old = self.mu.copy() 
            dist = [np.sum((X - self.mu[j,:])**2, axis = 1) for j in range(0,k)]
            self.p = np.argmin(dist, axis = 0)
            self.mu[range(0,k),:] = [np.nan_to_num(X[self.p==i,:].mean(axis = 0)) for i in range(0, k)]

            if np.allclose(self.mu, mu_old, atol=epsilon):
                break

        return self.p, self.mu

    def calculate_inertia(self, X):
        return sum([((X[self.p==i,:]-self.mu[i,:])**2).sum() for i in range(self.k)])   

   