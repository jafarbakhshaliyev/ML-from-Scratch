import numpy as np

class LinearSVM:
    
    def fit(self, X, y, C,  regularization = 'l1', tol=1e-3, max_iter=100):

        X = np.array(X)
        y = np.array(y)
        if regularization == 'l1': U = C; D = 0.0
        else: U = 10**5; D = 0.5*C
        n, d = X.shape
        alpha = np.zeros(n)
        w = np.zeros(d)
        iter_ = 0
        while iter_ < max_iter:
            alpha_prev = np.copy(alpha)
            
            for i in range(n):
                
                alpha_hat = alpha[i]
                G = y[i] * np.dot(w, X[i]) - 1 + alpha[i] * D
                
                if alpha[i] == 0: PG = min(G, 0)
                elif alpha[i] == U: PG = max(G, 0)
                else: PG = G
                    
                if abs(PG) > tol:
                    
                    Q_hat =  np.dot(X[i], X[i])
                    alpha[i] = min(max(alpha[i] - G/Q_hat, 0), U)
                    w += (alpha[i] - alpha_hat) * y[i] * X[i]
                    
            if np.linalg.norm(alpha - alpha_prev) < tol:
                break  
            iter_ += 1
            
        self.w = w    
        return self.w
    
    def predict(self, X):

        y_pred = np.sign(np.dot(X, self.w))
        
        return y_pred    