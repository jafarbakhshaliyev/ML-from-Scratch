import numpy as np 

class StandardScaler: 

    def fit_transform(self,x):

        self.mean = np.mean(x, axis = 0)
        self.std = np.std(x, axis = 0)
        x_norm = (x - self.mean)/(self.std)
        x_norm = np.nan_to_num(x_norm)
        return x_norm

    def transform(self, x):
        x_norm = (x - self.mean)/(self.std)
        x_norm = np.nan_to_num(x_norm)
        return x_norm