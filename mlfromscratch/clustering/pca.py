import numpy as np

class PCA:

    def fit(self, X, n):

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pcs = U @ np.diag(S)[:, :n]
        
        return pcs