import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import theano
import seaborn as sns

def main():



    #Read radon dataset
    data = pd.read_csv('radon.csv')

    #Collect unique counties and assign them an index
    county_names = data.county.unique()
    county_idx = data['county_code'].values

    #We are interested in whether having a basement (floor==0) or not (floor==1) increases radon measured
    #Measurement pooling - one single regression model
    #Separate regressions - for each county have an individual regression model
        #Model implies that there are no similarities between counties, which is unsatisfying
    #Hierarchical regression
        #While alpha and beta are different for each country, the coefficients come from a common group distribution

    '''
    #Separate bayesian regressions
    indiv_traces = {}
    for county_name in county_names:
        # Select subset of data belonging to county
        c_data = data.ix[data.county == county_name]
        c_data = c_data.reset_index(drop=True)
        c_log_radon = c_data.log_radon
        c_floor_measure = c_data.floor.values

        with pm.Model() as individual_model:
            # Intercept prior
            a = pm.Normal('alpha', mu=0, sd=100 ** 2)
            # Slope prior
            b = pm.Normal('beta', mu=0, sd=100 ** 2)
            # Model error prior as uniform distribution
            eps = pm.Uniform('eps', lower=0, upper=100)
            # Linear model
            radon_est = a + b * c_floor_measure
            # Data likelihood
            y_like = pm.Normal('y_like', mu=radon_est, sd=eps, observed=c_log_radon)
            # Inference
            step = pm.NUTS()
            trace = pm.sample(2000, step=step)
        indiv_traces[county_name] = trace
        pm.traceplot(trace)
    plt.show()
    '''

    #Hierarchical model
    #Initiates group parameters that consider counties not as completely different, but having an underlying similarity
    with pm.Model() as hierarchical_model:
        # Hyperpriors
        mu_a = pm.Normal('mu_alpha', mu=0., sd=100 ** 2)
        sigma_a = pm.Uniform('sigma_alpha', lower=0, upper=100)
        mu_b = pm.Normal('mu_beta', mu=0., sd=100 ** 2)
        sigma_b = pm.Uniform('sigma_beta', lower=0, upper=100)

        # Intercept for each county, distributed around group mean mu_a - radon levels
        a = pm.Normal('intercept', mu=mu_a, sd=sigma_a, shape=len(data.county.unique()))
        # Slope for each county, distributed around group mean mu_b - basement based
        b = pm.Normal('slope', mu=mu_b, sd=sigma_b, shape=len(data.county.unique()))

        # Model error
        eps = pm.Uniform('eps', lower=0, upper=100)

        # Expected value
        radon_est = a[county_idx] + b[county_idx] * data.floor.values

        # Data likelihood
        y_like = pm.Normal('y_like', mu=radon_est, sd=eps, observed=data.log_radon)
    with hierarchical_model:
        # Use ADVI for initialization
        mu, sds, elbo = pm.variational.advi(n=100000)
        step = pm.NUTS(scaling=hierarchical_model.dict_to_array(sds) ** 2,
                       is_cov=True)
        hierarchical_trace = pm.sample(5000, step, start=mu)
    pm.traceplot(hierarchical_trace[500:])
    plt.show()


    print('x')










if __name__ == '__main__':
    main()