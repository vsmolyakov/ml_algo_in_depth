import numpy as np 
import matplotlib.pyplot as plt 

from scipy.integrate import quad
from scipy.stats import multivariate_normal

np.random.seed(42)

class importance_sampler:
    # E[f(x)] = int_x f(x)p(x)dx = int_x f(x)[p(x)/q(x)]q(x) dx
    #         = sum_i f(x_i)w(x_i), where x_i ~ q(x)
    # e.g. for f(x) = 1(x \in A), E[f(x)] = P(A)
    
    def __init__(self, k=1.5, mu=0.8, sigma=np.sqrt(1.5), c=3):
        #target params p(x)
        self.k = k 

        #proposal params q(x)
        self.mu = mu 
        self.sigma = sigma 
        self.c = c #fix c, s.t. p(x) < c q(x)

    def target_pdf(self, x):
        #p(x) ~ Chi(k=1.5)
        return (x**(self.k-1)) * np.exp(-x**2/2.0)

    def proposal_pdf(self, x):
        #q(x) ~ N(mu,sigma)
        return self.c * 1.0/np.sqrt(2*np.pi*1.5) * np.exp(-(x-self.mu)**2/(2*self.sigma**2))
    
    def fx(self, x):
        #function of interest f(x), x >= 0
        return 2*np.sin((np.pi/1.5)*x)

    def sample(self, num_samples):    
        #sample from the proposal
        x = multivariate_normal.rvs(self.mu, self.sigma, num_samples)

        #discard netgative samples (since f(x) is defined for x >= 0)
        idx = np.where(x >= 0)
        x_pos = x[idx]

        #compute importance weights
        isw = self.target_pdf(x_pos) / self.proposal_pdf(x_pos)
        
        #compute E[f(x)] = sum_i f(x_i)w(x_i), where x_i ~ q(x)
        fw = (isw/np.sum(isw))*self.fx(x_pos)
        f_est = np.sum(fw)
    
        return isw, f_est
        

if __name__ == "__main__":

    num_samples = [10, 100, 1000, 10000, 100000, 1000000]

    F_est_iter, IS_weights_var_iter = [], []
    for k in num_samples:
        IS = importance_sampler()
        IS_weights, F_est = IS.sample(k)
        IS_weights_var = np.var(IS_weights/np.sum(IS_weights))
        F_est_iter.append(F_est)
        IS_weights_var_iter.append(IS_weights_var)

    #ground truth (numerical integration)
    k = 1.5
    I_gt, _ = quad(lambda x: 2.0*np.sin((np.pi/1.5)*x)*(x**(k-1))*np.exp(-x**2/2.0), 0, 5)

    #generate plots
    plt.figure()
    xx = np.linspace(0,8,100)
    plt.plot(xx, IS.target_pdf(xx), '-r', label='target pdf p(x)')
    plt.plot(xx, IS.proposal_pdf(xx), '--b', label='proposal pdf q(x)')
    plt.plot(xx, IS.fx(xx) * IS.target_pdf(xx), ':k', label='p(x)f(x) integrand')
    plt.grid(True); plt.legend(); plt.xlabel("X1"); plt.ylabel("X2")
    plt.title("Importance Sampling Components")
    #plt.savefig('./figures/importance_sampling.png')
    plt.show()

    plt.figure()
    plt.hist(IS_weights, label = "IS weights")
    plt.grid(True); plt.legend();
    plt.title("Importance Weights Histogram")
    #plt.savefig('./figures/importance_weights.png')
    plt.show()

    plt.figure()
    plt.semilogx(num_samples, F_est_iter, '-b', label = "IS Estimate of E[f(x)]")
    plt.semilogx(num_samples, I_gt*np.ones(len(num_samples)), '--r', label = "Ground Truth")
    plt.grid(True); plt.legend(); plt.xlabel('iterations'); plt.ylabel("E[f(x)] estimate")
    plt.title("IS Estimate of E[f(x)]")
    #plt.savefig('./figures/importance_estimate.png')
    plt.show()
