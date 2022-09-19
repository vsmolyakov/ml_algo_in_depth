import numpy as np 
import matplotlib.pyplot as plt 

import itertools
from numpy.linalg import inv
from scipy.stats import multivariate_normal

np.random.seed(42)

class gibbs_gauss:

    def gauss_conditional(self, mu, Sigma, setA, x):
        #computes P(X_A | X_B = x) = N(mu_{A|B}, Sigma_{A|B})
        dim = len(mu)
        setU = set(range(dim))    
        setB = setU.difference(setA)
        muA = np.array([mu[item] for item in setA]).reshape(-1,1)
        muB = np.array([mu[item] for item in setB]).reshape(-1,1)
        xB = np.array([x[item] for item in setB]).reshape(-1,1)
                    
        Sigma_AA = []
        for (idx1, idx2) in itertools.product(setA, setA):
            Sigma_AA.append(Sigma[idx1][idx2])
        Sigma_AA = np.array(Sigma_AA).reshape(len(setA),len(setA))

        Sigma_AB = []
        for (idx1, idx2) in itertools.product(setA, setB):
            Sigma_AB.append(Sigma[idx1][idx2])
        Sigma_AB = np.array(Sigma_AB).reshape(len(setA),len(setB))

        Sigma_BB = []
        for (idx1, idx2) in itertools.product(setB, setB):
            Sigma_BB.append(Sigma[idx1][idx2])
        Sigma_BB = np.array(Sigma_BB).reshape(len(setB),len(setB))

        Sigma_BB_inv = inv(Sigma_BB)
        mu_AgivenB = muA + np.matmul(np.matmul(Sigma_AB, Sigma_BB_inv), xB - muB)
        Sigma_AgivenB = Sigma_AA - np.matmul(np.matmul(Sigma_AB, Sigma_BB_inv), np.transpose(Sigma_AB))        

        return mu_AgivenB, Sigma_AgivenB

    def sample(self, mu, Sigma, xinit, num_samples):
        dim = len(mu)
        samples = np.zeros((num_samples, dim))
        x = xinit
        for s in range(num_samples):
            for d in range(dim):
                mu_AgivenB, Sigma_AgivenB = self.gauss_conditional(mu, Sigma, set([d]), x)
                x[d] = np.random.normal(mu_AgivenB, np.sqrt(Sigma_AgivenB))
            #end for
            samples[s,:] = np.transpose(x)
        #end for
        return samples

if __name__ == "__main__":

    num_samples = 2000
    mu = [1, 1]
    Sigma = [[2,1], [1,1]]    
    xinit = np.random.rand(len(mu),1)
    num_burnin = 1000

    gg = gibbs_gauss()
    gibbs_samples = gg.sample(mu, Sigma, xinit, num_samples)

    scipy_samples = multivariate_normal.rvs(mean=mu, cov=Sigma, size=num_samples, random_state=42)

    plt.figure()
    plt.scatter(gibbs_samples[num_burnin:,0], gibbs_samples[num_burnin:,1], c = 'blue', marker='s', alpha=0.8, label='Gibbs Samples')
    plt.scatter(scipy_samples[num_burnin:,0], scipy_samples[num_burnin:,1], c = 'red', alpha=0.8, label='Ground Truth Samples')
    plt.grid(True); plt.legend(); plt.xlim([-4,5])
    plt.title("Gibbs Sampling of Multivariate Gaussian"); plt.xlabel("X1"); plt.ylabel("X2")
    #plt.savefig("./figures/gibbs_gauss.png")
    plt.show()
    

