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
    #data = data.loc[data['county_code'].isin(['CASS', 'CROW WING', 'FREEBORN'])]

    #Collect unique counties and assign them an index
    county_names = data.county.unique()
    county_names2 = ['CASS', 'CROW WING', 'FREEBORN']
    county_idx = data['county_code'].values

    #We are interested in whether having a basement (floor==0) or not (floor==1) increases radon measured
    #Measurement pooling - one single regression model
    #Separate regressions - for each county have an individual regression model
        #Model implies that there are no similarities between counties, which is unsatisfying
    #Hierarchical regression
        #While alpha and beta are different for each country, the coefficients come from a common group distribution


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
            a = pm.Normal('intercept', mu=0, sd=100 ** 2)
            # Slope prior
            b = pm.Normal('slope', mu=0, sd=100 ** 2)
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
    #mu_a shows the mean radon levels, whily mu_b shows that having no basement decreases radon levels
    #Marginals for the intercept show that there are differences between counties


    #Posterior predictive check - recreating the data based on the parameters found at different moments in the chain
    #The recreated values are compared to real data points
    ppc = pm.sample_ppc(hierarchical_trace, model=hierarchical_model)
    #Create figure for the whole model
    plt.figure()
    ax = plt.subplot()
    sns.distplot([n.mean() for n in ppc['y_like']], kde=False, ax=ax)
    ax.axvline(data.log_radon.mean())
    ax.set(title='Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency')

    #Create ppc for 3 individual counties
    plt.figure()
    selection = ['CASS', 'CROW WING', 'FREEBORN']
    fig, axis = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
    axis = axis.ravel()
    for i, c in enumerate(selection):
        c_data = data.ix[data.county == c]
        c_data = c_data.reset_index(drop=True)
        z = list(c_data['county_code'])[0]

        xvals = np.linspace(-0.2, 1.2)
        #Plot sample traces for the individual model
        for a_val, b_val in zip(indiv_traces[c]['intercept'][500::10], indiv_traces[c]['slope'][500::10]):
            axis[i].plot(xvals, a_val + b_val * xvals, 'b', alpha=.1)
        #Plot mean trace for the individual model
        axis[i].plot(xvals, indiv_traces[c]['intercept'][500::10].mean() + indiv_traces[c]['slope'][500::10].mean() * xvals,
                     'b', alpha=1, lw=2., label='individual')
        #Plot sample traces for the hierarchical model
        for a_val, b_val in zip(hierarchical_trace['intercept'][500::10][z], hierarchical_trace['slope'][500::10][z]):
            axis[i].plot(xvals, a_val + b_val * xvals, 'g', alpha=.1)
        #Plot mean trace for the hierarchical model
        axis[i].plot(xvals, hierarchical_trace['intercept'][500::10][z].mean() + hierarchical_trace['slope'][500::10][
            z].mean() * xvals,
                     'g', alpha=1, lw=2., label='hierarchical')
        #Plot datapoints
        axis[i].scatter(c_data.floor + np.random.randn(len(c_data)) * 0.01, c_data.log_radon,
                        alpha=1, color='k', marker='.', s=80, label='original data')
        axis[i].set_xticks([0, 1])
        axis[i].set_xticklabels(['basement', 'no basement'])
        axis[i].set_ylim(-1, 4)
        axis[i].set_title(c)
        if not i % 3:
            axis[i].legend()
            axis[i].set_ylabel('log radon level')

    #Shrinkage - estimates are pulled towards the group mean
    #County coefficients very far away from the group mean have very low probability under normality assumption
    hier_a = hierarchical_trace['intercept'][500:].mean(axis=0)
    hier_b = hierarchical_trace['slope'][500:].mean(axis=0)
    indv_a = [indiv_traces[c]['intercept'][500:].mean() for c in county_names]
    indv_b = [indiv_traces[c]['slope'][500:].mean() for c in county_names]
    #Plot arrows to show how a hierarchical model influences the posterior mean
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, xlabel='Intercept', ylabel='Floor Measure',
                         title='Hierarchical vs. Non-hierarchical Bayes',
                         xlim=(0, 3), ylim=(-3, 3))

    ax.scatter(indv_a, indv_b, s=26, alpha=0.4, label='non-hierarchical')
    ax.scatter(hier_a, hier_b, c='red', s=26, alpha=0.4, label='hierarchical')
    for i in range(len(indv_b)):
        ax.arrow(indv_a[i], indv_b[i], hier_a[i] - indv_a[i], hier_b[i] - indv_b[i],
                 fc="k", ec="k", length_includes_head=True, alpha=0.4, head_width=.04)
    ax.legend()



    plt.show()


    print('x')










if __name__ == '__main__':
    main()