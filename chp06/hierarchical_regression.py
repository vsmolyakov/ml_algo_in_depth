import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import pymc3 as pm 

def main():
    
    #load data
    data = pd.read_csv('./data/radon.txt')
    
    county_names = data.county.unique()
    county_idx = data['county_code'].values
    
    with pm.Model() as hierarchical_model:
        
        # Hyperpriors
        mu_a = pm.Normal('mu_alpha', mu=0., sd=100**2)
        sigma_a = pm.Uniform('sigma_alpha', lower=0, upper=100)
        mu_b = pm.Normal('mu_beta', mu=0., sd=100**2)
        sigma_b = pm.Uniform('sigma_beta', lower=0, upper=100)
    
        # Intercept for each county, distributed around group mean mu_a
        a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(data.county.unique()))
        # Slope for each county, distributed around group mean mu_b
        b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(data.county.unique()))
    
        # Model error
        eps = pm.Uniform('eps', lower=0, upper=100)
    
        # Expected value
        radon_est = a[county_idx] + b[county_idx] * data.floor.values
    
        # Data likelihood
        y_like = pm.Normal('y_like', mu=radon_est, sd=eps, observed=data.log_radon)
        
    
    with hierarchical_model:
        # Use ADVI for initialization
        mu, sds, elbo = pm.variational.advi(n=100000)
        step = pm.NUTS(scaling=hierarchical_model.dict_to_array(sds)**2, is_cov=True)
        hierarchical_trace = pm.sample(5000, step, start=mu)

                
    pm.traceplot(hierarchical_trace[500:])
    plt.show()        
        
if __name__ == "__main__":    
    main()
