import numpy as np
import matplotlib.pyplot as plt

np.random.seed(14)

class KDE():

    def __init__(self):
        #Histogram and Gaussian Kernel Estimator used to
        #analyze RNA-seq data for flux estimation of a T7 promoter
        self.G = 1e9  #length of genome in base pairs (bp)
        self.C = 1e3  #number of unique molecules
        self.L = 100  #length of a read, bp
        self.N = 1e6  #number of reads, L bp long
        self.M = 1e4  #number of unique read sequences, bp
        self.LN = 1000  #total length of assembled / mapped RNA-seq reads
        self.FDR = 0.05  #false discovery rate

        #uniform sampling (poisson model)
        self.lmbda = (self.N * self.L) / self.G  #expected number of bases covered
        self.C_est = self.M/(1-np.exp(-self.lmbda)) #library size estimate
        self.C_cvrg = self.G - self.G * np.exp(-self.lmbda) #base coverage
        self.N_gaps = self.N * np.exp(-self.lmbda) #number of gaps (uncovered bases)

        #gamma prior sampling (negative binomial model)
        #X = "number of failures before rth success"
        self.k = 0.5 # dispersion parameter (fit to data)
        self.p = self.lmbda/(self.lmbda + 1/self.k) # success probability
        self.r = 1/self.k # number of successes
        
        #RNAP binding data (RNA-seq)
        self.data = np.random.negative_binomial(self.r, self.p, size=self.LN)

    def histogram(self):
        self.bin_delta = 1  #smoothing parameter 
        self.bin_range = np.arange(1, np.max(self.data), self.bin_delta)
        self.bin_counts, _ = np.histogram(self.data, bins=self.bin_range)

        #histogram density estimation 
        #P = integral_R p(x) dx, where X is in R^3
        #p(x) = K/(NxV), where K=number of points in region R
        #N=total number of points, V=volume of region R

        rnap_density_est = self.bin_counts/(sum(self.bin_counts) * self.bin_delta)
        return rnap_density_est

    def kernel(self):
        #Gaussian kernel density estimator with smoothing parameter h
        #sum N Guassians centered at each data point, parameterized by common std dev h

        x_dim = 1  #dimension of x
        h = 10 #standard deviation

        rnap_density_support = np.arange(np.max(self.data))
        rnap_density_est = 0
        for i in range(np.sum(self.bin_counts)):
            rnap_density_est += (1/(2*np.pi*h**2)**(x_dim/2.0))*np.exp(-(rnap_density_support - self.data[i])**2 / (2.0*h**2))
        #end for
        
        rnap_density_est = rnap_density_est / np.sum(rnap_density_est)
        return rnap_density_est

if __name__ == "__main__":

    kde = KDE()
    est1 = kde.histogram()
    est2 = kde.kernel()

    plt.figure()
    plt.plot(est1, '-b', label='histogram')
    plt.plot(est2, '--r', label='gaussian kernel')
    plt.title("RNA-seq density estimate based on negative binomial model")
    plt.xlabel("read length, [base pairs]"); plt.ylabel("density"); plt.legend()
    plt.show()