import math
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

class ridge_reg():

    def __init__(self, n_iter=20, learning_rate=1e-3, lmbda=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.lmbda = lmbda 
    
    def fit(self, X, y):
        #insert const 1 for bias term
        X = np.insert(X, 0, 1, axis=1)
        
        self.loss = []
        self.w = np.random.rand(X.shape[1])

        for i in range(self.n_iter):
            y_pred = X.dot(self.w)
            mse = np.mean(0.5*(y - y_pred)**2 + 0.5*self.lmbda*self.w.T.dot(self.w))
            self.loss.append(mse)
            print(" %d iter, mse: %.4f" %(i, mse))
            #compute gradient of NLL(w) wrt w
            grad_w = - (y - y_pred).dot(X) + self.lmbda*self.w
            #update the weights
            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        #insert const 1 for bias term
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

if __name__ == "__main__":
    
    X, y = load_boston(return_X_y=True)
    X_reg = X[:,6].reshape(-1,1) #house age 
    X_std = (X_reg - X_reg.mean())/X.std() #standard scaling
    y_std = (y - y.mean())/y.std() #standard scaling

    rr = ridge_reg()
    rr.fit(X_std, y_std)
    y_pred = rr.predict(X_std)

    print(rr.w)

    plt.figure()
    plt.plot(rr.loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(X_std, y_std)
    plt.plot(np.linspace(-1,1), rr.w[1]*np.linspace(-1,1)+rr.w[0], c='red')
    plt.xlim([-0.5,0.23])
    plt.xlabel("scaled house age")
    plt.ylabel("house price")
    plt.show()