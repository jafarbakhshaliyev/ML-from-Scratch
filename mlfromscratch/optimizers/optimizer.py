import numpy as np 

class StepLength:

    def bolddriver(self, fnew, flast, mu, mplus = 1.1, mminus = 0.5):

        if fnew < flast:

            mu = mu*mplus

        else:
            mu = mu*mminus

        return mu

    def RMSProp(self, mu, v, gradient, beta = 0.5):

        v = beta*v + (1-beta)*gradient**2
        mu = mu/(np.sqrt(v + 1e-08))

        return mu, v