import numpy as np

class NaiveBayes:
    
    def fit(self, X, y, alpha = 1.0):
        
        self.X = X
        self._classes = np.unique(y)
        self.n_class = len(np.unique(y))
        self.size_instance, self.size_feature = X.shape
        self.cond_probs, self.class_prior  = np.zeros((self.size_feature, self.n_class)), {}

        for cl in range(self.n_class):
            
            X_cl = X[y == cl]
            self.cond_probs[:,cl] = self.conditional_probs(X_cl, alpha)
            self.class_prior[str(cl)] = X_cl.shape[0] / self.size_instance
            
        return self
    
    def predict(self, X):
        return [self.predict_one(xi) for xi in X]
    
    def predict_one(self, xi):
        
        posterior = []
        
        for cl in range(self.n_class):
            
            prior = np.log(self.class_prior[str(cl)])
            likelihoods_c = self.calc_likelihood(self.cond_probs[:,cl].reshape(1,-1), xi.toarray())
            posteriors_c = np.sum(likelihoods_c) + prior
            posterior.append(posteriors_c)
            
        return self._classes[np.argmax(posterior)]  
            
    def calc_likelihood(self, cls_likeli, xi):
        return np.multiply(np.log(cls_likeli),xi)
  
    def conditional_probs(self, X, alpha):
        
        cond_prob = (np.sum(X, axis = 0) + alpha)/(np.sum(X) + alpha*self.X.shape[1])
        
        return cond_prob                                                         