import numpy as np
from numpy.linalg import norm

np.random.seed(42)

class page_rank():

    def __init__(self):
        self.max_iter = 100
        self.tolerance = 1e-5
    
    def power_iteration(self, A):
        n = np.shape(A)[0]
        v = np.random.rand(n)
        converged = False
        iter = 0

        while (not converged) and (iter < self.max_iter):
            old_v = v
            v = np.dot(A, v)
            v = v / norm(v)
            lambd = np.dot(v, np.dot(A, v))
            converged = norm(v - old_v) < self.tolerance
            iter += 1
        #end while

        return lambd, v
    
if __name__ == "__main__":

    #construct a symmetric real matrix
    X = np.random.rand(10,5)
    A = np.dot(X.T, X)
    
    pr = page_rank()
    lambd, v = pr.power_iteration(A)
    
    print(lambd)
    print(v)

    #compare against np.linalg implementation
    eigval, eigvec = np.linalg.eig(A)
    idx = np.argsort(np.abs(eigval))[::-1]
    top_lambd = eigval[idx][0]
    top_v = eigvec[:,idx][0]    

    assert np.allclose(lambd, top_lambd, 1e-3)
    assert np.allclose(v, top_v, 1e-3)




