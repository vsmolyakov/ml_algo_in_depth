import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def pi_est():
    
    radius = 1
    num_iter = int(1e4)
    
    X = np.random.uniform(-radius,+radius,num_iter)
    Y = np.random.uniform(-radius,+radius,num_iter)
    
    R2 = X**2 + Y**2
    inside = R2 < radius**2
    outside = ~inside
    
    samples = (2*radius)*(2*radius)*inside
    
    I_hat = np.mean(samples)
    pi_hat = I_hat/radius ** 2
    pi_hat_se = np.std(samples)/np.sqrt(num_iter)    
    print("pi est: {} +/- {:f}".format(pi_hat, pi_hat_se))
    
    plt.figure()
    plt.scatter(X[inside],Y[inside], c='b', alpha=0.5)
    plt.scatter(X[outside],Y[outside], c='r', alpha=0.5)
    plt.show()
    
if __name__ == "__main__":
    
    pi_est()
    
    