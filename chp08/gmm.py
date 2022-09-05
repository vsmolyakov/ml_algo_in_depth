import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from scipy import linalg

np.random.seed(3)

class GMM:

    def __init__(self, n=1e3, d=2, K=4):
        self.n = int(n)  #number of data points
        self.d = d  #data dimension 
        self.K = K  #number of clusters

        self.X = np.zeros((self.n, self.d))

        self.mu = np.zeros((self.d, self.K))
        self.sigma = np.zeros((self.d, self.d, self.K))
        self.pik = np.ones(self.K)/K

    def generate_data(self):
        #GMM generative model
        alpha0 = np.ones(self.K) 
        pi = np.random.dirichlet(alpha0)

        #ground truth mu and sigma
        mu0 = np.random.randint(0, 10, size=(self.d, self.K)) - 5*np.ones((self.d, self.K))
        V0 = np.zeros((self.d, self.d, self.K))
        for k in range(self.K):
            eigen_mean = 0 
            Q = np.random.normal(loc=0, scale=1, size=(self.d, self.d))
            D = np.diag(abs(eigen_mean + np.random.normal(loc=0, scale=1, size=self.d)))
            V0[:,:,k] = abs(np.transpose(Q)*D*Q)
        
        #sample data
        for i in range(self.n):
            z = np.random.multinomial(1,pi)
            k = np.nonzero(z)[0][0]
            self.X[i,:] = np.random.multivariate_normal(mean=mu0[:,k], cov=V0[:,:,k], size=1)

        plt.figure()
        plt.scatter(self.X[:,0], self.X[:,1], color='b', alpha=0.5)
        plt.title("Ground Truth Data"); plt.xlabel("X1"); plt.ylabel("X2")
        plt.show()

        return mu0, V0

    def gmm_em(self):
        
        #init mu with k-means
        kmeans = KMeans(n_clusters=self.K, random_state=42).fit(self.X)
        self.mu = np.transpose(kmeans.cluster_centers_)

        #init sigma
        for k in range(self.K):
            self.sigma[:,:,k] = np.eye(self.d)

        #EM algorithm
        max_iter = 10
        tol = 1e-5
        obj = np.zeros(max_iter)
        for iter in range(max_iter):
            print("EM iter ", iter)
            #E-step
            resp, llh = self.estep()
            #M-step
            self.mstep(resp)
            #check convergence
            obj[iter] = llh
            if (iter > 1 and obj[iter] - obj[iter-1] < tol*abs(obj[iter])):
                break
            #end if
        #end for
        plt.figure()
        plt.plot(obj)
        plt.title('EM-GMM objective'); plt.xlabel("iter"); plt.ylabel("log-likelihood")
        plt.show()

    def estep(self):
        
        log_r = np.zeros((self.n, self.K))
        for k in range(self.K):
            log_r[:,k] = multivariate_normal.logpdf(self.X, mean=self.mu[:,k], cov=self.sigma[:,:,k])
        #end for
        log_r = log_r + np.log(self.pik)
        L = logsumexp(log_r, axis=1)
        llh = np.sum(L)/self.n  #log likelihood
        log_r = log_r - L.reshape(-1,1) #normalize
        resp = np.exp(log_r)
        return resp, llh

    def mstep(self, resp):

        nk = np.sum(resp, axis=0)
        self.pik = nk/self.n
        sqrt_resp = np.sqrt(resp)
        for k in range(self.K):
            #update mu
            rx = np.multiply(resp[:,k].reshape(-1,1), self.X)
            self.mu[:,k] = np.sum(rx, axis=0) / nk[k]

            #update sigma
            Xm = self.X - self.mu[:,k]
            Xm = np.multiply(sqrt_resp[:,k].reshape(-1,1), Xm)
            self.sigma[:,:,k] = np.maximum(0, np.dot(np.transpose(Xm), Xm) / nk[k] + 1e-5 * np.eye(self.d))
        #end for

if __name__ == '__main__':

    gmm = GMM()
    mu0, V0 = gmm.generate_data()
    gmm.gmm_em()
    
    for k in range(mu0.shape[1]):
        print("cluster ", k)
        print("-----------")
        print("ground truth means:")
        print(mu0[:,k])
        print("ground truth covariance:")
        print(V0[:,:,k])
    #end for 
    
    for k in range(mu0.shape[1]):
        print("cluster ", k)
        print("-----------")
        print("GMM-EM means:")
        print(gmm.mu[:,k])
        print("GMM-EM covariance:")
        print(gmm.sigma[:,:,k])

    plt.figure()
    ax = plt.axes()
    plt.scatter(gmm.X[:,0], gmm.X[:,1], color='b', alpha=0.5)

    for k in range(mu0.shape[1]):

        v, w = linalg.eigh(gmm.sigma[:,:,k])
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        # plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(gmm.mu[:,k], v[0], v[1], 180.0 + angle, color='r', alpha=0.5)
        ax.add_patch(ell)

        # plot cluster centroids
        plt.scatter(gmm.mu[0,k], gmm.mu[1,k], s=80, marker='x', color='k', alpha=1)
    plt.title("Gaussian Mixture Model"); plt.xlabel("X1"); plt.ylabel("X2")
    plt.show()
