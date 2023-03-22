import numpy as np 
from losses.loss import Loss

class SGD:
    
    def __init__(self,X,y,loss, reg, lambd, ratio = 0.5):
        self.X = X
        self.y = y
        self.loss = loss
        self.reg = reg
        self.lambd = lambd
        self.ratio = ratio
            
    def min_sgd(self, theta0, mu, C, K, show_epoch = False):
        
        theta = theta0
        if self.reg not in ["L1", "L2", "ElasticNet"]:
            reg = False
        
        logloss_old, _, _ = self.loss(self, self.X, self.y, theta)
        if reg == True:
            regloss, _, _ = self.reg(self, theta, self.lambd, self.ratio)
            logloss_old += regloss

        self.J_hist = np.zeros((K+1,1))
        self.J_hist[0] = logloss_old
        self.theta_val = np.zeros((K,self.X.shape[1]))
        
        for i in range(K):
            
            x_train, y_train = self.shuffle_data(self.X, self.y, C)
            
            for j in range(C):

                xvar = x_train[j,:].reshape(1,self.X.shape[1])
                yvar = y_train[j].reshape(1, 1)
                losses, gradient, _ = self.loss(self, xvar, yvar, theta)
                if reg == True:
                    losses, gradient, _ = tuple(x + y for x, y in zip(self.loss(self, xvar, yvar, theta), self.reg(self, theta, self.lambd, self.ratio)))
                theta = theta - mu*gradient
            
                
            logloss_new, _, _ =  self.loss(self, x_train, y_train, theta)
            if reg == True:
                regloss, _, _ = self.reg(self, theta, self.lambd, self.ratio)
                logloss_new += regloss
            
            logloss_old = logloss_new
            self.J_hist[i+1] = logloss_new
            self.theta_val[i] = theta.reshape(1,self.X.shape[1])

            if show_epoch == True:
                print("Epoch %d | logloss %f" % (i, losses))

        
        return self.J_hist, self.theta_val, theta
    
    def shuffle_data(self, x,y, C):
        
        index = np.random.randint(len(self.y),size = C)
        np.random.shuffle(index)
        x = x[index,:]
        y = y[index]
        
        return x, y
    