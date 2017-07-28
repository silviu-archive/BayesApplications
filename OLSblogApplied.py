import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style("darkgrid")
import patsy as pt
import statsmodels.api as sm
import pymc3 as pm
from scipy.optimize import fmin_powell


rndst = np.random.RandomState(0)


def generateData(n=20, a=1, b=1, c=0, latent_error_y=10):
    #Create dataset based on linear model - noisy physical process
    #From: y ~ a + bx +cx^2 + e

    df = pd.DataFrame({'x':rndst.choice(np.arange(100), n, replace=False)})
    df['y'] = a + b * (df['x']) + c * (df['x']) ** 2

    #Add noise based on normal distribution
    df['y'] += rndst.normal(0, latent_error_y, n)

    return df

def plotPosteriorCr(mdl, trc, rawdata, xlims, npoints=1000):
    #Plot the posterior predictions from model given traces

    #Extract traces - samples that were collected
    trc_mu = pm.trace_to_dataframe(trc)[['Intercept', 'x']]
    trc_sd = pm.trace_to_dataframe(trc)['sd']

    #Recreate the likelihood
    x = np.linspace(xlims[0], xlims[1], npoints).reshape((npoints,1))
    X = x ** np.ones((npoints,2)) * np.arange(2)
    like_mu = np.dot(X,trc_mu.T)
    like_sd = np.tile(trc_sd.T,(npoints,1))
    like = np.random.normal(like_mu, like_sd)

    #Calculate credible regions and plot over the datapoints
    dfp = pd.DataFrame(np.percentile(like, [2.5, 25, 50, 75, 97.5], axis=1).T
                       , columns=['025', '250', '500', '750', '975'])
    dfp['x'] = x

    pal = sns.color_palette('Purples')
    f, ax1d = plt.subplots(1, 1, figsize=(7, 7))
    ax1d.fill_between(dfp['x'], dfp['025'], dfp['975'], alpha=0.5
                      , color=pal[1], label='CR 95%')
    ax1d.fill_between(dfp['x'], dfp['250'], dfp['750'], alpha=0.4
                      , color=pal[4], label='CR 50%')
    ax1d.plot(dfp['x'], dfp['500'], alpha=0.5, color=pal[5], label='Median')
    _ = plt.legend()
    _ = ax1d.set_xlim(xlims)
    _ = sns.regplot(x='x', y='y', data=rawdata, fit_reg=False
                    , scatter_kws={'alpha': 0.8, 's': 80, 'lw': 2, 'edgecolor': 'w'}, ax=ax1d)




def main():

    df = generateData(a=5, b=2, latent_error_y=30)
    #Parameters beta are [5, 2]
    #Variance is 30

    g = sns.lmplot(x='x', y='y', data=df, fit_reg=True, size=6,
                   scatter_kws={'alpha':0.8, 's':60})


    #Encode model specification as design matrices
    fml = 'y ~ 1 + x'
    (mx_en, mx_ex) = pt.dmatrices(fml, df, return_type='dataframe', NA_action='raise')

    #Fit OLS model
    smfit = sm.OLS(endog=mx_en,exog=mx_ex, hasconst=True).fit()
    print(smfit.summary())

    #Model specifications are wrapped in a with-statement
    with pm.Model() as mdl_ols:

        #Use GLM submodule for simplified patsy-like model spec
        #Use Normal family - normal distribution likelihood, HalfCauchy distribution priors
        pm.glm.GLM.from_formula('y ~ 1 + x', df, family=pm.glm.families.Normal())

        #Find MAP(maximum a posterior) using Powell optimization
        #Mode of the posterior distribution
        start_MAP = pm.find_MAP(fmin=fmin_powell, disp=True)

        #Take samples using NUTS from the joint probability distribution
        #Iteratively converges by minimising loss on posterior predictive distribution yhat with respect to true y
        trc_ols = pm.sample(2000, start=start_MAP, step=pm.NUTS())


    ax = pm.traceplot(trc_ols[-1000:], figsize=(12, len(trc_ols.varnames)*1.5),
                      lines = {k: v['mean'] for k, v in pm.df_summary(trc_ols[-1000:]).iterrows()})

    print(pm.df_summary(trc_ols[-1000:]))

    xlims = (df['x'].min() - np.ptp(df['x']) / 10
             , df['x'].max() + np.ptp(df['x']) / 10)

    plotPosteriorCr(mdl_ols, trc_ols, df, xlims)



    plt.show()

    print('x')



if __name__ == '__main__':
    main()


