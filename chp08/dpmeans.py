import numpy as np
import matplotlib.pyplot as plt

import time
from sklearn import metrics
from sklearn.datasets import load_iris

np.random.seed(42)

class dpmeans:
    
    def __init__(self,X):
        # Initialize parameters for DP means
        self.K = 1
        self.K_init = 4
        self.d = X.shape[1]
        self.z = np.mod(np.random.permutation(X.shape[0]),self.K)+1
        self.mu = np.random.standard_normal((self.K, self.d))
        self.sigma = 1
        self.nk = np.zeros(self.K)
        self.pik = np.ones(self.K)/self.K 
        
        #init mu
        self.mu = np.array([np.mean(X,0)])
        
        #init lambda
        self.Lambda = self.kpp_init(X,self.K_init)
        
        self.max_iter = 100
        self.obj = np.zeros(self.max_iter)
        self.em_time = np.zeros(self.max_iter)   
        
    def kpp_init(self,X,k):
        #k++ init
        #lambda is max distance to k++ means

        [n,d] = np.shape(X)
        mu = np.zeros((k,d))        
        dist = np.inf*np.ones(n)
            
        mu[0,:] = X[int(np.random.rand()*n-1),:]
        for i in range(1,k):
            D = X-np.tile(mu[i-1,:],(n,1))
            dist = np.minimum(dist, np.sum(D*D,1))
            idx = np.where(np.random.rand() < np.cumsum(dist/float(sum(dist))))
            mu[i,:] = X[idx[0][0],:]
            Lambda = np.max(dist)
        
        print("Lambda: ", Lambda)
        
        return Lambda
        
    def fit(self,X):

        obj_tol = 1e-3
        max_iter = self.max_iter        
        [n,d] = np.shape(X)
        
        obj = np.zeros(max_iter)
        em_time = np.zeros(max_iter)
        print('running dpmeans...')
        
        for iter in range(max_iter):
            tic = time.time()
            dist = np.zeros((n,self.K))
            
            #assignment step
            for kk in range(self.K):
                Xm = X - np.tile(self.mu[kk,:],(n,1))
                dist[:,kk] = np.sum(Xm*Xm,1)
            
            #update labels
            dmin = np.min(dist,1)
            self.z = np.argmin(dist,1)
            idx = np.where(dmin > self.Lambda)
            
            if (np.size(idx) > 0):
                self.K = self.K + 1
                self.z[idx[0]] = self.K-1 #cluster labels in [0,...,K-1]
                self.mu = np.vstack([self.mu,np.mean(X[idx[0],:],0)])                
                Xm = X - np.tile(self.mu[self.K-1,:],(n,1))
                dist = np.hstack([dist, np.array([np.sum(Xm*Xm,1)]).T])
             
            #update step
            self.nk = np.zeros(self.K)
            for kk in range(self.K):
                self.nk[kk] = self.z.tolist().count(kk)
                idx = np.where(self.z == kk)
                self.mu[kk,:] = np.mean(X[idx[0],:],0)
            
            self.pik = self.nk/float(np.sum(self.nk))
            
            #compute objective
            for kk in range(self.K):
                idx = np.where(self.z == kk)
                obj[iter] = obj[iter] + np.sum(dist[idx[0],kk],0)                
            obj[iter] = obj[iter] + self.Lambda * self.K
            
            #check convergence
            if (iter > 0 and np.abs(obj[iter]-obj[iter-1]) < obj_tol*obj[iter]):
                print('converged in %d iterations\n'% iter)
                break
            em_time[iter] = time.time()-tic
        #end for
        self.obj = obj
        self.em_time = em_time
        return self.z, obj, em_time
        
    def compute_nmi(self, z1, z2):
        # compute normalized mutual information
        
        n = np.size(z1)
        k1 = np.size(np.unique(z1))
        k2 = np.size(np.unique(z2))
        
        nk1 = np.zeros((k1,1))
        nk2 = np.zeros((k2,1))

        for kk in range(k1):
            nk1[kk] = np.sum(z1==kk)
        for kk in range(k2):
            nk2[kk] = np.sum(z2==kk)
            
        pk1 = nk1/float(np.sum(nk1))
        pk2 = nk2/float(np.sum(nk2))
        
        nk12 = np.zeros((k1,k2))
        for ii in range(k1):
            for jj in range(k2):
                nk12[ii,jj] = np.sum((z1==ii)*(z2==jj))
        pk12 = nk12/float(n)        
        
        Hx = -np.sum(pk1 * np.log(pk1 + np.finfo(float).eps))
        Hy = -np.sum(pk2 * np.log(pk2 + np.finfo(float).eps))
        
        Hxy = -np.sum(pk12 * np.log(pk12 + np.finfo(float).eps))
        
        MI = Hx + Hy - Hxy;
        nmi = MI/float(0.5*(Hx+Hy))
        
        return nmi

    def generate_plots(self,X):

        plt.close('all')
        plt.figure(0)
        for kk in range(self.K):
            #idx = np.where(self.z == kk)
            plt.scatter(X[self.z == kk,0], X[self.z == kk,1], \
                        s = 100, marker = 'o', c = np.random.rand(3,), label = str(kk))
        #end for
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.title('DP-means clusters')
        plt.grid(True)
        plt.show()
        
        plt.figure(1)
        plt.plot(self.obj)
        plt.title('DP-means objective function')
        plt.xlabel('iterations')
        plt.ylabel('penalized l2 squared distance')
        plt.grid(True)
        plt.show()
        
if __name__ == "__main__":        
    
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    dp = dpmeans(X)
    labels, obj, em_time = dp.fit(X)
    dp.generate_plots(X)

    nmi = dp.compute_nmi(y,labels)
    ari = metrics.adjusted_rand_score(y,labels)
    
    print("NMI: %.4f" % nmi)
    print("ARI: %.4f" % ari)       
