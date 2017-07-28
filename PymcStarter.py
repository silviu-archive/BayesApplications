import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import scipy.stats as stats
import theano


def main():
    ##Coin toss example
    '''
    #Binomial likelihood, Beta prior
    #Number of tosses
    n = 100
    #Number of times it was heads
    h = 61
    #Distribution hyperparameters
    alpha = 2
    beta = 2
    #Number of iterations
    nIter = 1000

    #Context management for pymc3
    with pm.Model() as model:
        #Define prior distribution
        p = pm.Beta('p', alpha=alpha, beta=beta)
        #Define likelihood [p(y|theta)] as Binomial(n,p)
        #n = number of Bernoulli trials
        #p = probability of success in each trial
        y = pm.Binomial('y', n=n, p=p, observed=h)
        #Inferencing
        #Find Maximum A Posteriori estimate (via optimization) as the initial state for MCMC
        start = pm.find_MAP()
        #Choose a sampler
        step = pm.Metropolis()
        #Return the trace from the sampling procedure
        trace = pm.sample(nIter, step, start, random_seed=123, progressbar=True)

    #Plot the sample extracted from the posterior distribution as a histogram
    plt.hist(trace['p'], 15, histtype='step', normed=True, label='Posterior')
    #Create array for x-coordinates
    x = np.linspace(0, 1, 100)
    #Plot a beta distribution B(alpha, beta) representing the prior [p(theta)]
    plt.plot(x, stats.beta.pdf(x, alpha, beta), label='Prior')
    #Add legend
    plt.legend()
    '''

    '''
    ##Estimating mean and stdev of normal distribution
    #Number of samples
    N = 100
    #Normal distribution mean
    _mu = np.array([10])
    #Normal distribution standard deviation
    _sigma = np.array([2])
    #N samples from the normal distribution with mean _mu and stdev _sigma
    y = np.random.normal(_mu, _sigma, N)
    
    nIter = 1000
    
    with pm.Model() as model:
        #Define prior distribution's mean and standard deviation as sampled from a uniform distribution
        mu = pm.Uniform('mu', lower=0, upper=100, shape=_mu.shape)
        sigma = pm.Uniform('sigma', lower=0, upper=10, shape=_sigma.shape)
        #Define likelihood from a normal distribution based on mu and sigma
        #The observed argument indicates that the values were observed, and should not be changed by any fitting algorithm
        y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)
    
        #Inferencing procedure to determine posterior samples
        start = pm.find_MAP()
        step = pm.Slice()
        trace = pm.sample(nIter, step, start, random_seed=123, progressbar=True)
    
    
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    #Plot histogram of the latter half of posterior mu
    #Latter half are selected due to sampling burn-in (not reaching convergence)
    plt.hist(trace['mu'][-nIter/2:,0], 25, histtype='step')
    plt.subplot(1,2,2)
    #Plot histogram of the latter half of posterior sigma
    plt.hist(trace['sigma'][-nIter/2:,0], 25, histtype='step')
    '''

    ##Estimating parameters of a linear regression model
    # y ~ ax + b
    # y = ax + b + e
    # y ~ N(ax + b, sigma^2)
    #Assume priors (t = precision = 1 / sigma^2):
    #a ~ N(0, 100)
    #b ~ N(0, 100)
    #t ~ Gamma(0.1, 0.1)

    #Number of Iterations
    nIter = 1000
    #Observed dat
    #Number of observations
    n = 11
    #Coefficients
    _a = 6
    _b = 2
    #Uniform distribution of x
    x = np.linspace(0, 1, n)
    #Linear model (including random error)
    y = _a*x + _b + np.random.randn(n)

    with pm.Model() as model:
        #Determine distributions from which to sample the coefficients
        a = pm.Normal('a', mu=0, sd=20)
        b = pm.Normal('b', mu=0, sd=20)
        sigma = pm.Uniform('sigma', lower=0, upper=20)

        #Estimate mean of likelihood
        y_est = a*x + b

        #Likelihood
        likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)

        #Inference
        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(nIter, step, start, random_seed=123, progressbar=True)
        #Plot traceplot for a, b, and stdev
        pm.traceplot(trace)


    #Alternative formulation using General Linear Models
    data = dict(x=x, y=y)

    with pm.Model() as model:
        pm.glm.GLM.from_formula('y ~ x', data)
        step = pm.NUTS()
        trace = pm.sample(2000, step, progressbar=True)
        pm.traceplot(trace)

    #Plot regression line and datapoints
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=30, label='data')
    pm.plot_posterior_predictive_glm(trace, samples=100,
                                     label='posterior predictive regression lines',
                                     c='blue', alpha=0.2)
    plt.plot(x, _a*x + _b, label='true regression line', lw=3., c='red')
    plt.legend(loc='best')
    #Fit is skewed due to low levell of mass in the tails of the normal distribution
    #An outlier affects the fit strongly

    #We can assume that the data is not normally distributed, but distributed according to Student T distribution
    #Comparing the two distributions
    normal_dist = pm.Normal.dist(mu=0, sd=1)
    t_dist = pm.StudentT.dist(mu=0, lam=1, nu=1)
    x_eval = np.linspace(-8, 8, 300)
    plt.figure()
    plt.plot(x_eval, theano.tensor.exp(normal_dist.logp(x_eval)).eval(), label='Normal', lw=2.)
    plt.plot(x_eval, theano.tensor.exp(t_dist.logp(x_eval)).eval(), label='Student T', lw=2.)
    plt.xlabel('x')
    plt.ylabel('Probability density')
    plt.legend()

    #Testing the model with a student T from the families object
    with pm.Model() as model_robust:
        family = pm.glm.families.StudentT()
        pm.glm.GLM.from_formula('y ~ x', data, family=family)
        trace_robust = pm.sample(2000)
        pm.traceplot(trace_robust)

    #Plotting the model specifying our data is spread according to a student T distribution
    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=30, label='data')
    pm.plot_posterior_predictive_glm(trace_robust,
                                     label='posterior predictive regression lines')
    plt.plot(x, _a*x + _b, label='true regression line', lw=3., c='red')
    plt.legend()




    plt.show()

    print('x')

if __name__ == '__main__':
    main()
