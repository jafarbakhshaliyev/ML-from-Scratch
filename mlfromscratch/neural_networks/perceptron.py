import numpy as np

class Perceptron:
    
    def activation(self, z):
        
        return np.where(z>0, 1, -1) 
    
    def accuracy(self, y_true, y_pred):
        
        accuracy = np.sum(y_true == y_pred) # not dividing by len because we have one
        
        return accuracy
    
    def fit(self, X, y, lr, epochs):
        
        m, n = X.shape
        theta = np.zeros((n+1,1))
        theta_val = np.zeros((epochs,n+1))
        
        for epoch in range(epochs):
            
            accuracy = 0
            
            for i, x_i in enumerate(X):
                
                x_i = np.insert(x_i, 0, 1).reshape(-1,1) 
                y_hat = self.activation(np.dot(x_i.T, theta)) 
              
                accuracy += self.accuracy(y[i],y_hat) #
                
                if (np.squeeze(y_hat) - y[i]) != 0: 
                    theta += lr*((y[i] - y_hat)/2*x_i)
        
            accuracy /= m 
            theta_val[epoch] = theta.reshape(3)

            
            print('Epoch {}/{}'.format(epoch, epochs - 1),':', 'Train Accuracy: {:.4f}'.format(accuracy)) 
            self.theta = theta 
            self.theta_val = theta_val
            
            if accuracy == 1.0: 
                return self.theta
        
        return self.theta 
    
    def predict(self, X):
        
        
        X = np.hstack((np.ones((X.shape[0],1)),X)) # adding intercept
        self.pred = self.activation(np.dot(X, self.theta).reshape(1,-1)) 
        
        return self.pred
