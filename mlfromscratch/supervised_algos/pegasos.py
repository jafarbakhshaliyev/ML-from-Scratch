import numpy as np

class Pegasos:
    
    def fit(self, X, y, lambd, T, k = 1, projection = False):
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        for t in range(1, T+1):
            batch_indices = np.random.choice(n_samples, size = k, replace = False)
            X_batch = np.array(X.iloc[batch_indices])
            y_batch = y[batch_indices]
            n = 1 / (lambd * t)
            gradient = np.zeros(n_features)
            for i in range(k):
                if y_batch.iloc[i] * np.dot(self.w, X_batch[i]) < 1:
                    gradient += y_batch.iloc[i] * X_batch[i]
                    
            self.w = (1-n*lambd)*self.w + (n/k)*gradient
            if projection: self.w = min(1, 1 / (np.sqrt(lambd) * np.linalg.norm(self.w)))*self.w
                
        return self.w 
    
    def predict(self, X):
        
        y_pred = np.sign(np.dot(X, self.w))
        
        return y_pred 