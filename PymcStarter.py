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
    #Linear model (including random error) with n observations
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
        #Coefficients are fixed
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
    '''


    '''
    #Read pandas dataset with sex, heights, and weights
    df = pd.read_csv('FakeHeight.csv')
    #Number of iterations
    nIter = 1000
    with pm.Model() as model:
        #Is a person a male based on height + weight, assuming that the data is distributed according to a Binomial
        pm.glm.GLM.from_formula('male ~ height + weight', df)
        #Height and Weight's coefficients are now traced
        trace = pm.sample(nIter, step=pm.Slice(), random_seed=123, progressbar=True)
        pm.traceplot(trace)

    #Plot scatter matrix between intercept, height and weight
    df_trace = pm.trace_to_dataframe(trace)
    pd.scatter_matrix(df_trace[-1000:], diagonal='kde')

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(df_trace.ix[-1000:, 'height'], linewidth=0.7)
    plt.subplot(122)
    plt.plot(df_trace.ix[-1000:, 'weight'], linewidth=0.7)

    pm.summary(trace)
    #As a conclusion, when height's coefficient decreases, weight's coefficient increases
    #Plots suggest a strong correlation between the 2 variables


    #Attempting the same results with a different sampler
    with pm.Model() as model:
        pm.glm.GLM.from_formula('male ~ height + weight', df, family=pm.glm.families.Binomial())
        trace = pm.sample(nIter, step=pm.NUTS(), random_seed=123, progressbar=True)

    df_trace = pm.trace_to_dataframe(trace)
    pd.scatter_matrix(df_trace[-1000:], diagonal='kde')

    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(df_trace.ix[-1000:, 'height'], linewidth=0.7)
    plt.subplot(122)
    plt.plot(df_trace.ix[-1000:, 'weight'], linewidth=0.7)

    pm.summary(trace)

    #KDE plot in seaborn
    plt.figure()
    sns.kdeplot(trace['weight'], trace['height'])
    plt.xlabel('Weight', fontsize=20)
    plt.ylabel('Height', fontsize=20)
    plt.style.use('ggplot')




    #Select intercept, height, weight means from the latter half of the trace
    intercept, height, weight = df_trace[-nIter // 2:].mean(0)

    def predict(w, h, height=height, weight=weight):
        #Predict gender given weight (w) and height (h) values
        v = intercept + height * h + weight * w
        return np.exp(v) / (1 + np.exp(v))

    #Calculate predictions on grid
    #Define uniform weights
    xs = np.linspace(df.weight.min(), df.weight.max(), 100)
    #Define uniform heights
    ys = np.linspace(df.height.min(), df.height.max(), 100)
    #Merge the 2 arrays
    X, Y = np.meshgrid(xs, ys)
    #Predict gender where X = weight, Y = height
    Z = predict(X, Y)

    plt.figure(figsize=(6, 6))
    #Plot 0.5 contour line - classify as male if above this line
    plt.contour(X, Y, Z, levels=[0.5])
    #Classify all subjects
    colors = ['lime' if i else 'yellow' for i in df.male]
    #Create predictions on existing data
    ps = predict(df.weight, df.height)
    #Create errors
    errs = ((ps < 0.5) & df.male) | ((ps >= 0.5) & (1 - df.male))
    #Plot error circles
    plt.scatter(df.weight[errs], df.height[errs], facecolors='red', s=150)
    #Plot data points
    plt.scatter(df.weight, df.height, facecolors=colors, edgecolors='k', s=50, alpha=1)
    plt.xlabel('Weight', fontsize=16)
    plt.ylabel('Height', fontsize=16)
    plt.title('Gender classification by weight and height', fontsize=16)
    plt.tight_layout()
    '''

    '''
    #Estimating parameters of a logistic model (example from Gelman et al.)
    #Model where dose of a drug may be affect the number of rat deaths in an experiment
    #Model number of deaths as a random sample from a binomial distribution, n = no. of rats, p = probability of death
    #n = 5, but p may be related to drug dose x
    #As x increases, the number of dying rats seems to increase
    #y ~ Binomial (n, p)
    #logit(p) = a + bx
    #a ~ N(0, 5)
    #b ~ N(0, 10)
    #a and b have vague priors (as parameters for the model)

    nIter = 1000
    #Observed data
    n = 5 * np.ones(4)
    x = np.array([-0.896, -0.296, -0.053, 0.727])
    y = np.array([0, 1, 3, 5])

    def invlogit(x):
        return np.exp(x) / (1 + np.exp(x))

    with pm.Model() as model:
        #Define priors
        alpha = pm.Normal('alpha', mu=0, sd=5)
        #beta = pm.Flat('beta')
        beta = pm.Normal('beta', mu=0, sd=10)

        #Define likelihood [p(y|theta)] as Binomial(n,p)
        #n = number of trials
        #p = probability of success in each trial
        p = invlogit(alpha + beta*x)
        y_obs = pm.Binomial('y_obs', n=n, p=p, observed=y)


        #Inference
        start = pm.find_MAP()
        step = pm.NUTS()
        trace = pm.sample(nIter, step, start, random_seed=123, progressbar=True)
        pm.traceplot(trace)

    pm.summary(trace)

    f = lambda a, b, xp: np.exp(a + b * xp) / (1 + np.exp(a + b * xp))
    xp = np.linspace(-1, 1, 100)
    a = trace.get_values('alpha').mean()
    b = trace.get_values('beta').mean()
    plt.figure()
    plt.plot(xp, f(a, b, xp))
    plt.scatter(x, y / 5, s=50)
    plt.xlabel('Log does of drug')
    plt.ylabel('Risk of death')
    '''


    #Using a hierarchical model based on Radon data set
    radon = pd.read_csv('radon.csv')[['county', 'floor', 'log_radon']]
    radon.dropna(inplace=True)

    #With a hierarchical model, there is an ac and bc for each county c just as an individual county model ac*x + b = y
    #They are no longer independent but assume to come from a common group distribution
    #ac ~ N(mua, sda^2)
    #bc ~ N(mub, sdb^2)
    #Further assumptions for the hyperparameters
    #mua ~ N(0, 100^2)
    #sda ~ U(0, 100)
    #mub ~ N(0, 100^2)
    #sdb ~ U(0, 100)

    #Transform counties into codes
    county = pd.Categorical(radon['county']).codes

    with pm.Model() as hm:
        #County hyperparameters
        mu_a = pm.Normal('mu_a', mu=0, tau=1.0/100**2)
        sigma_a = pm.Uniform('sigma_a', lower=0, upper=100)
        mu_b = pm.Normal('mu_b', mu=0, tau=1.0/100 ** 2)
        sigma_b = pm.Uniform('sigma_b', lower=0, upper=100)

        #County slope and intercept - can either provide sd or tau
        a = pm.Normal('slope', mu=mu_a, sd=sigma_a, shape=len(set(county)))
        b = pm.Normal('intercept', mu=mu_b,  tau = 1.0/sigma_b**2, shape=len(set(county)))

        #Household errors - following gamma distribution
        sigma = pm.Gamma('sigma', alpha=10, beta=1)

        #Model prediction of radon level based on county distribution of floor levels
        mu = a[county] + b[county] * radon.floor.values

        #Likelihood
        y = pm.Normal('y', mu=mu, sd=sigma, observed=radon.log_radon)

    with hm:
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        hm_trace = pm.sample(2000, step, start=start, random_seed=123, progressbar=True)

    pm.summary(hm_trace)

    plt.figure(figsize=(8, 60))
    pm.forestplot(hm_trace, varnames=['slope', 'intercept'])


    plt.show()

    print('x')

if __name__ == '__main__':
    main()
