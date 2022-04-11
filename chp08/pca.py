import numpy as np

class PCA():
    def __init__(self, n_components = 2):
        self.n_components = n_components

    def covariance_matrix(self, X, Y=None):
        if Y is None:
            Y = X
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
        return covariance_matrix
    
    def transform(self, X):
        Sigma = self.covariance_matrix(X)
        eig_vals, eig_vecs = np.linalg.eig(Sigma)

        #sort from largest to smallest and select the first n_components
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx][:self.n_components]
        eig_vecs = np.atleast_1d(eig_vecs[:,idx])[:, :self.n_components]

        #project the data onto principal components
        X_transformed = X.dot(eig_vecs)

        return X_transformed

if __name__ == "__main__":

    n = 10
    d = 5
    X = np.random.rand(n,d)

    pca = PCA(n_components = 2)
    X_pca = pca.transform(X)

    print(X_pca)
