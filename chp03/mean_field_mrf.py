import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from scipy.special import expit as sigmoid
from scipy.stats import multivariate_normal

np.random.seed(42)
sns.set_style('whitegrid')

class image_denoising:

    def __init__(self, img_binary, sigma=2, J=1):

        #mean-field parameters
        self.sigma  = sigma  #noise level
        self.y = img_binary + self.sigma*np.random.randn(M, N) #y_i ~ N(x_i; sigma^2);
        self.J = J  #coupling strength (w_ij)
        self.rate = 0.5  #update smoothing rate
        self.max_iter = 15
        self.ELBO = np.zeros(self.max_iter)
        self.Hx_mean = np.zeros(self.max_iter)

    def mean_field(self):

        #Mean-Field VI
        print("running mean-field variational inference...")
        logodds = multivariate_normal.logpdf(self.y.flatten(), mean=+1, cov=self.sigma**2) - \
                  multivariate_normal.logpdf(self.y.flatten(), mean=-1, cov=self.sigma**2)
        logodds = np.reshape(logodds, (M, N))

        #init
        p1 = sigmoid(logodds)
        mu = 2*p1-1  #mu_init

        a = mu + 0.5 * logodds
        qxp1 = sigmoid(+2*a)  #q_i(x_i=+1)
        qxm1 = sigmoid(-2*a)  #q_i(x_i=-1)

        logp1 = np.reshape(multivariate_normal.logpdf(self.y.flatten(), mean=+1, cov=self.sigma**2), (M, N))
        logm1 = np.reshape(multivariate_normal.logpdf(self.y.flatten(), mean=-1, cov=self.sigma**2), (M, N))

        for i in tqdm(range(self.max_iter)):
            muNew = mu
            for ix in range(N):
                for iy in range(M):
                    pos = iy + M*ix
                    neighborhood = pos + np.array([-1,1,-M,M])            
                    boundary_idx = [iy!=0,iy!=M-1,ix!=0,ix!=N-1]
                    neighborhood = neighborhood[np.where(boundary_idx)[0]]            
                    xx, yy = np.unravel_index(pos, (M,N), order='F')
                    nx, ny = np.unravel_index(neighborhood, (M,N), order='F')
            
                    Sbar = self.J*np.sum(mu[nx,ny])       
                    muNew[xx,yy] = (1-self.rate)*muNew[xx,yy] + self.rate*np.tanh(Sbar + 0.5*logodds[xx,yy])
                    self.ELBO[i] = self.ELBO[i] + 0.5*(Sbar * muNew[xx,yy])
                #end for
            #end for
            mu = muNew
            
            a = mu + 0.5 * logodds
            qxp1 = sigmoid(+2*a) #q_i(x_i=+1)
            qxm1 = sigmoid(-2*a) #q_i(x_i=-1)    
            Hx = -qxm1*np.log(qxm1+1e-10) - qxp1*np.log(qxp1+1e-10) #entropy        
    
            self.ELBO[i] = self.ELBO[i] + np.sum(qxp1*logp1 + qxm1*logm1) + np.sum(Hx)
            self.Hx_mean[i] = np.mean(Hx)            
        #end for
        return mu

if __name__ == "__main__":

    #load data
    print("loading data...")
    data = Image.open('./figures/bayes.bmp')
    img = np.double(data)
    img_mean = np.mean(img)
    img_binary = +1*(img>img_mean) + -1*(img<img_mean)
    [M, N] = img_binary.shape

    mrf = image_denoising(img_binary, sigma=2, J=1)
    mu = mrf.mean_field()

    #generate plots
    plt.figure()
    plt.imshow(mrf.y)
    plt.title("observed noisy image")
    #plt.savefig('./figures/ising_vi_observed_image.png')
    plt.show()

    plt.figure()
    plt.imshow(mu)
    plt.title("after %d mean-field iterations" %mrf.max_iter)
    #plt.savefig('./figures/ising_vi_denoised_image.png')
    plt.show()

    plt.figure()
    plt.plot(mrf.Hx_mean, color='b', lw=2.0, label='Avg Entropy')
    plt.title('Variational Inference for Ising Model')
    plt.xlabel('iterations'); plt.ylabel('average entropy')
    plt.legend(loc='upper right')
    #plt.savefig('./figures/ising_vi_avg_entropy.png')
    plt.show()

    plt.figure()
    plt.plot(mrf.ELBO, color='b', lw=2.0, label='ELBO')
    plt.title('Variational Inference for Ising Model')
    plt.xlabel('iterations'); plt.ylabel('ELBO objective')
    plt.legend(loc='upper left')
    #plt.savefig('./figures/ising_vi_elbo.png')
    plt.show()

