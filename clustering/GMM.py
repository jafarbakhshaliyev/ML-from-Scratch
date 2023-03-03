import numpy as np
import matplotlib.pyplot as plt

class GaussianMixtureModel:

    def fit(self, X, k, maxiter = 100, epsilon = 10e-5):
        
        N, M = X.shape
        self.k = k
        pi, mu = np.zeros(k), np.zeros((k,M))
        S = [np.eye(M)] * k
        q = np.random.uniform(0, 1, size=(N, k))
        q = q/np.sum(q, axis = 1)[:, np.newaxis]


        for iter_ in range(maxiter):
            q_old = q.copy()
            
            # Maximization
            for j in range(k):
                pi[j] = np.sum(q, axis = 0)[j] / N
                mu[j] = (q[:, j].T@X)/ np.sum(q, axis = 0)[j]
                S[j] = ((q[:,j] * ((X - mu[j]).T)) @ (X - mu[j])) / np.sum(q, axis = 0)[j]
                
            # Expectation
            for j in range(k):
                q[:, j] = pi[j] * self._multivariate_normal.pdf(X, mu[j], S[j])    
                
            q = q / np.sum(q, axis = 1)[:, np.newaxis]

            if np.allclose(q, q_old, atol = epsilon):
                break

        self.mu = mu
        self.S = S  

        return self

    def predict(self, X, prob = False):

        probs = np.array([self._multivariate_normal.pdf(X, mean = self.mu[j], cov = self.S[j]) for j in range(self.k) ])

        if prob == False: return np.argmax(probs, axis = 0) 
        
        return probs
    
    def _multivariate_normal(self, X, mean, cov):

        L = np.linalg.cholesky(cov)
        cf = np.prod(np.diag(L)) / (2 * np.pi) ** (mean.size / 2)
        z = np.linalg.solve(L, X - mean)
        
        return cf * np.exp(-0.5 * z.dot(z))
