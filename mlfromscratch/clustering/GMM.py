import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GaussianMixtureModel:

    def fit(self, X, k, maxiter = 100, epsilon = 10e-5):
        
        N, M = X.shape
        self.k = k
        pi, self.mu = np.zeros(k), np.zeros((k,M))
        self.S = [np.eye(M)] * k
        q = np.random.uniform(0, 1, size=(N, k))
        q = q/np.sum(q, axis = 1)[:, np.newaxis]


        for iter_ in range(maxiter):
            q_old = q.copy()
            
            # Maximization
            for j in range(k):
                pi[j] = np.sum(q, axis = 0)[j] / N
                self.mu[j] = (q[:, j].T@X)/ np.sum(q, axis = 0)[j]
                self.S[j] = ((q[:,j] * ((X - self.mu[j]).T)) @ (X - self.mu[j])) / np.sum(q, axis = 0)[j]
                
            # Expectation
            for j in range(k):
                q[:, j] = pi[j] * multivariate_normal.pdf(X, self.mu[j], self.S[j])    
                
            q = q / np.sum(q, axis = 1)[:, np.newaxis]

            if np.allclose(q, q_old, atol = epsilon):
                break

        return self

    def predict(self, X, prob = False):

        probs = np.array([multivariate_normal.pdf(X, mean = self.mu[j], cov = self.S[j]) for j in range(self.k) ])

        if prob == False: return np.argmax(probs, axis = 0) 
        
        return probs
