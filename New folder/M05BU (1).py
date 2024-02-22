# THERE IS UPDATED CCD class !!!

# Package installation
import subprocess as _spr
import sys as _sys

def install(package_name: str) -> None:
    try:
        exec('import ' + package_name)
    except ModuleNotFoundError:
        _spr.run(
            [_sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True,
            text=True)

install('dexpy')
install('probscale')
install('pyDOE')

from pyDOE import *

from itertools import chain, combinations, combinations_with_replacement, product
from math import ceil
from typing import List

import dexpy
import dexpy.alias                                                          # function to generate confounded factors
from dexpy.ccd import build_ccd
import dexpy.factorial
from dexpy.model import make_model, ModelOrder
from dexpy.optimal import build_optimal
from typing import Union as _Union
from IPython.display import Latex, Markdown
from IPython.display import Markdown as _Markdown
from IPython.display import display as _display
import matplotlib.pyplot as plt
import numpy as np
_np = np
import pandas as _pd
from patsy import dmatrix
import probscale
import scipy
from scipy.interpolate import interp1d
from scipy.special import comb
import scipy.stats as sps
import seaborn as sns
from sklearn import preprocessing
import statsmodels.api as sm
import statsmodels as _sm           # AIE-AIE-AIE
from statsmodels.graphics.api import interaction_plot
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
from tabulate import tabulate
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.optimize import _check_optimize_result
from scipy.optimize import minimize


plt.rcParams['axes.grid'] = True                # for resid_fourpack, but applied everywhere

def closest_node(node, nodes):
  nodes = np.asarray(nodes)
  dist_2 = np.sum((nodes - node)**2, axis=1)
  return np.argmin(dist_2)

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx

def mk3Dplot_arrays(X, n=50):
    x = np.linspace(X[:,0].min(), X[:,0].max(), n)
    y = np.linspace(X[:,1].min(), X[:,1].max(), n)
    xy = np.array(list(product(x, y)))
    return (
        xy,
        xy[:, 0].reshape(n, n),
        xy[:, 1].reshape(n, n)
    )

class MyGPR(GaussianProcessRegressor):
    def __init__(self, kernel, n_restarts_optimizer=15, max_iter=2e05, gtol=1e-06):
        super(MyGPR, self).__init__(kernel=kernel, n_restarts_optimizer=n_restarts_optimizer)
        self.max_iter = max_iter
        self.gtol = gtol

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self.max_iter, 'gtol': self.gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

def print_columns(df):
    print('List of columns for identification:')
    for i, col in enumerate(df.columns):
        print(i, col, sep='   ')


def print_variables(dictionaries):
    print('Variables Dictionaries:\n')
    for dictionary in dictionaries:
        print(
            tabulate(
                [[key, val] for key, val in dictionary.items()],
                headers=['variable', 'column']
            ),
            end='\n\n'
        )


def display_model_coefficient_table(model, input_df):
    model_coeff_df = _pd.DataFrame()
    model_coeff_df['coef']    = model.params
    model_coeff_df['std err'] = model.bse
    model_coeff_df['t']       = model.tvalues
    model_coeff_df['p-value'] = model.pvalues
    model_coeff_df['VIF']     = [
        variance_inflation_factor(input_df.values, i)
        for i, _
        in enumerate(input_df.columns)
    ]

    print('Model Coefficients', end='\n\n')
    display(model_coeff_df.round(3))


def display_model_evaluation_table(model):
    data = {
        'R2':     [model.rsquared],
        'R2-adj': [model.rsquared_adj],
        'PRESS':  [OLSInfluence(model).ess_press],
        'AIC':    [model.aic],
        'BIC':    [model.bic]
    }

    model_eval_df = _pd.DataFrame(data, columns=list(data.keys()))
    model_eval_df.index=[' ']

    print('Model Summary', end='\n\n')
    display(model_eval_df.round(3))


def printMarkdown(s: str) -> None:
    """
    Outputs markdown from a code cell, e.g.

    printMarkdown('#Heading')
    """
    _display(_Markdown(''.join(s.splitlines())))


def sigfigs(x, n):
    if x == 0:
        y = x
    else:
        y = round(
            x,
            -int(_np.floor(_np.log10(abs(x)))) + (n - 1)
        )

    return y


def RegResults(
    model: _sm.regression.linear_model.RegressionResultsWrapper
) -> _pd.DataFrame:
    press = OLSInfluence(model).ess_press
    press_rmse = _np.sqrt(press / model.nobs)
    df = _pd.DataFrame(
        index=[''],
        columns=['R-squared', 'F-statistic', 'p (F-statistic)',
                 'No. observations', 'Model df', 'Residual df',
                 'PRESS_RMSE'],
        data=[[model.rsquared, model.fvalue, round(model.f_pvalue, 2),
               model.nobs, model.df_model, model.df_resid, press_rmse]]
    )
    df = pd.sigfigs(df, 3)
    for col in ['No. observations', 'Model df', 'Residual df']:
        df[col] = df[col].astype(int)
  
    return df


def RegCoefficients(
    model: _sm.regression.linear_model.RegressionResultsWrapper,
    X: _pd.DataFrame,
    alpha = 0.1
) -> _pd.DataFrame:
    dfm = _pd.DataFrame()
    dfm['coef'] = model.params
    dfm[f'[{alpha / 2: .3f}'] = model.conf_int(alpha)[0]
    dfm[f'{1 - alpha / 2: .3f}]'] = model.conf_int(alpha)[1]
    dfm['std err'] = model.bse
    dfm['t'] = model.tvalues
    dfm['p-value'] = model.pvalues.round(2)
    dfm['VIF'] = [variance_inflation_factor(X.values, i)
                  for i, col in enumerate(model.params)]
    dfm = pd.sigfigs(dfm, 3)

    return dfm


def resid_fourpack(student_resid, fits):
    '''...'''
    fig, axes = plt.subplots(2, 2)              # creating subplots
    fig.set_size_inches(18, 12)                 # setting figure dimensions

    # Normal Probability Plot
    sps.probplot(student_resid, plot=axes[0, 0])
    axes[0, 0].set_title('Normal Probability Plot')

    # Deleted Residuals vs Fitted Value Plot
    axes[0, 1].plot(fits, student_resid, 'o', color='b')
    axes[0, 1].set_title('Deleted Residuals vs Fits')
    axes[0, 1].set_xlabel('Fitted Value')
    axes[0, 1].set_ylabel('Deleted Residuals')
    axes[0, 1].set_ylim(-3.5, 3.5)
    
    # Histogram
    sns.histplot(student_resid, stat='frequency', ax=axes[1, 0])
    axes[1, 0].set_title('Histogram')
    axes[1, 0].set_xlim(-3.5, 3.5)
    
    # Deleted Residuals vs Observation Order
    axes[1, 1].plot(student_resid, '-o')
    axes[1, 1].set_title('Deleted Residuals vs Order');
    axes[1, 1].set_xlabel('Observation Order')
    axes[1, 1].set_ylabel('Deleted Residuals')
    axes[1, 1].set_ylim(-3.5, 3.5)


class Stats:
    '''
    Class constructor

    Call class with any argument that will make a Pandas series

    e.g. s = stats[df['X']]
         s = stats([1.1, 2.0, 1.6])

    Any missing values will be ignored.
    '''
    def __init__(self, data, *, description='Stats class module',
                 varname='Data', alpha=0.05):
        # remove any missing values, avoid the bug with integer series
        self.X = _pd.Series(data).dropna().astype(float).sort_values().reset_index(drop=True)
        
        self.xbar = self.X.mean()               # sample mean
        self.median = self.X.median()           # sample median
        self.S = self.X.std()                   # sample standard deviation
        self.sem = self.X.sem()                 # standard error of the mean
        self.n = self.X.size                    # sample size

        self.dsc = description                  # descriptive string for output
        self.vrn = varname                      # descriptive variable name for output (string)
        
        self.alpha = alpha                      # alpha level for confidence intervals and tests
        
        self.meanci = self.__meanconf()
        self.medci = self.__medconf()
        self.stdci = self.__stdconf()


    def description(self, description): # CHANGE TO SETTER? OR IS IT EVEN NEEDED?
        'Change the description used for plots'
        self.dsc = description


    def varname(self, varname): # CHANGE TO SETTER? OR IS IT EVEN NEEDED?
        '''...'''
        self.vrn = varname


    def set_alpha(self, new_alpha):
        'Change the alpha level for tests etc.'
        self.alpha = new_alpha
        self.meanci = self.__meanconf()
        self.medci = self.__medconf()
        self.stdci = self.__stdconf()


    def __meanconf(self):
        'Calculate t-based confindence interval for the mean'
        return sps.t.interval(
            alpha=1 - self.alpha,
            loc=self.xbar,
            scale=self.sem,
            df=self.n - 1
        )


    def __stdconf(self):
        'Calculate chisq-based confidence interval for the variance'
        ci = sps.chi2.interval(
            alpha=1 - self.alpha,
            df=self.n - 1
        )
        lower = np.sqrt((self.n - 1) / ci[1]) * self.S
        upper = np.sqrt((self.n - 1) / ci[0]) * self.S
    
        return (lower, upper)
  

    def dstats(self):
        'Returns a Pandas series containing descriptive statistics for the data'
        sk = _pd.Series(
            {
                'skewness': self.X.skew(),
                'kurtosis': self.X.kurtosis()
            }
        )
        dser = self.X.describe()
        dser = dser.rename(index={'50%': 'median'})
        
        return _pd.concat([dser, sk])
  

    def ad(self):
        'Returns an Anderson-Darling normality test as a pandas series'
        andy = sps.anderson(self.X)
        fp = interp1d(andy.critical_values, andy.significance_level / 100, 
                      kind='linear', fill_value='extrapolate')
        Asq = andy.statistic
        p = fp(Asq).item()
        return _pd.Series({'A-Squared': Asq, 
                           'P-Value': p})


    def hist(self, autotitle=True, *args, **kwargs):
        'Takes the same keyword arguments as the Seaborn histplot routine'
        ax = sns.histplot(
            self.X,
            stat='density',
            kde=True,
            *args, **kwargs
        )
        if autotitle:
            ax.set_title(f'Histogram for {self.dsc}')
            ax.set_xlabel(self.vrn)
    
        return ax


    def __pp(self):
        '''
        Constructs plotting positions using Filiben's method

        Returns a Pandas series that contains the plotting positions
        (they do not depend on the data)
        '''
        out = np.zeros(self.n)
        out[0] = 1 - 0.5**(1 / self.n)
        for k in range(1, self.n - 1):
            out[k] = (k + 0.6825) / (self.n + 0.365)
        out[self.n - 1] = 1 - out[0]

        return _pd.Series(out)
  

    def xvar(self):
        '''
        Produces a linearly spaced series from mu - 3 sigma to mu + 3 sigma
        that is used in plotting empirical cdf's and probability plots
        '''
        out = np.linspace(
            self.xbar - 3 * self.S,
            self.xbar + 3 * self.S, 100
        )
    
        return _pd.Series(out)


    def __cdf(self, percent=True):
        out = sps.norm.cdf(self.xvar(), self.xbar, self.S)
        if percent:
            out = 100 * out

        return _pd.Series(out)
  

    def ecdf(self, *args, percent=True, **kwargs):
        Y = self.__pp()
        if percent:
            Y = 100 * Y
        ax = sns.scatterplot(x=self.X, y=Y, *args, **kwargs)
        sns.lineplot(x=self.xvar(), y=self.__cdf(percent=percent), color='red', ax=ax)
        ax.grid()
        ax.set_title(f'Empirical CDF: {self.dsc}')
        ax.set_xlabel(self.vrn)
        if percent:
            ax.set_ylabel('Cumulative probability [%]')
        else:
            ax.set_ylabel('Cumulative probability [-]')

        return ax


    def nplt(self, *args, **kwargs):
        ax = self.ecdf(*args, **kwargs)
        ax.set_title(f'Normal Probability Plot: {self.dsc}')
        ax.set_ylim(0.51, 99.49)
        ax.set_yscale('prob')
        ax.annotate(
            f'mean = {round(self.xbar, 3)}',
            (.05,.85),
            xycoords='axes fraction'
        )
        ax.annotate(
            f'std = {round(self.S, 3)}',
            (.05,.77),
            xycoords='axes fraction'
        )
        return ax
  

    def boxplot(self, autotitle=True, *args, **kwargs):
        ax = sns.boxplot(data=self.X, orient='h', *args, **kwargs)
        ax.axes.yaxis.set_visible(False)
        if autotitle:
            ax.set_title(f'Boxplot for {self.dsc}')
        
        return ax
  

    def __unifarr(self, nrows=10, mcols=10):
        return np.random.uniform(size=(nrows, mcols))
  

    def __kdeicdf(self, p):
        kde = sm.nonparametric.KDEUnivariate(self.X)
        kde.fit()
        Finv = interp1d(
            kde.cdf, kde.support,
            kind='cubic',
            fill_value='extrapolate'
        )
    
        return Finv(p)
  

    def __bootsample(self, nrows=10, mcols=10):
        p = self.__unifarr(nrows=nrows, mcols=mcols)
        x = self.__kdeicdf(p)
        
        return _pd.DataFrame(x)
  

    def __medconf(self, bootsize=10000):
        bs = self.__bootsample(nrows=self.n, mcols=bootsize)
        lower = bs.median().quantile(q=self.alpha / 2)
        upper = bs.median().quantile(q=1-self.alpha / 2)
        
        return (lower, upper)
  

    def summary(self):
        info = [
            self.dsc,
            'Anderson-Darling Normality Test',
            self.ad().to_string(float_format="{:9.3f}"), # ??? .format
            'Descriptive Statistics',
            self.dstats().to_string(float_format="{:10.3f}"),
            f'{1 - self.alpha:.0%} Confidence Interval',
            'For Mean:',
            f'{self.meanci[0]:10.3f}',
            f'{self.meanci[1]:10.3f}',
            'For Median',
            f'{self.medci[0]:10.3f}',
            f'{self.medci[1]:10.3f}',
            'For StDev',
            f'{self.stdci[0]:10.3f}',
            f'{self.stdci[1]:10.3f}'
        ]

        return '\n'.join(info)
  

    def gsummary(self, nplt=False):
        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = fig.add_gridspec(4, 3)
        f_ax1 = fig.add_subplot(gs[:2, :2])
        f_ax2 = fig.add_subplot(gs[2, :2])
        f_ax3 = fig.add_subplot(gs[3, :2])
        f_ax4 = fig.add_subplot(gs[:, 2])
        ax = f_ax1

        if nplt:
            ax = self.nplt(ax=ax)
        else:
            ax = self.hist(autotitle=False, ax=ax)

        ax.set_title(
            f'Graphical Summary for {self.dsc}',
            fontsize=16,
            fontweight='bold'
        )

        a, b = ax.get_xlim()
        ax = f_ax2
        ax = self.boxplot(autotitle=False, ax=ax)
        ax.set_xlim(a, b)
        ax.axes.xaxis.set_visible(False)

        ax = f_ax3
        bpdf = _pd.DataFrame() 
        bpdf['mean'] = [
            self.meanci[0],
            self.meanci[0],
            self.xbar,
            self.meanci[1],
            self.meanci[1]
        ]

        bpdf['median'] = [
            self.medci[0],
            self.medci[0],
            self.median,
            self.medci[1],
            self.medci[1]
        ]

        sns.boxplot(data=bpdf, ax=ax, orient='h')
        ax.set_title(
            f'{1 - self.alpha:.0%}  Confidence Intervals for Mean and Median',
            fontsize=12,
            fontweight='normal'
        )
        ax.set_xlim(a, b)
        f_ax4.axis('off')
        f_ax4.text(
            0.1, 0.2, self.summary(),
            fontsize=14, fontweight='bold', fontfamily='monospace'
        )

        return fig, f_ax1, f_ax2, f_ax3, f_ax4


def _rename_columns(df):
    '''...'''
    assert len(df.columns) <= 26, f'Not enough letters in English alphabet! {len(df.columns)}'
    col_names = {f'X{i + 1}': chr(65 + i) for i in range(len(df))}
    return df.rename(columns=col_names)


def _get_aliases(df, k):
    '''...'''
    assert k <= 26, 'Not enough letters in English alphabet!'
    descriptor = '+'.join([chr(65 + i) for i in range(k)])
    descriptor = f'({descriptor})**{k}' if k > 1 else descriptor

    aliases, _ = dexpy.alias.alias_list(descriptor, df)

    return aliases


def full_fact(k):
    '''...'''
    n = 2**k
    df = dexpy.factorial.build_factorial(k, n)
    df = _rename_columns(df)

    print(f'Experiment is two level full factorial with {k} factors, and {n} runs.', end='\n\n')
    print('Design matrix:', end='\n\n')
    display(df)

    return df


def frac_fact(k, p):
    m = 2**(k - p)
    df = dexpy.factorial.build_factorial(k, m)

    df = _rename_columns(df)
    aliases = _get_aliases(df, k)
  
    print(f'Experiment is 1 / {2**p} fraction {k} factors, in {m} runs.', end='\n\n')
    print('Design matrix', end='\n\n')
    display(df.head())
    print('\n\n\n')
    print('Alias Structure', end='\n\n')
    display(aliases)

    return df, aliases


def effect_plot(
    x_vars: List[str],
    y_var: List[str],
    df: _pd.DataFrame
):
    '''...'''
    g = sns.PairGrid(
        data=df,
        x_vars=x_vars,
        y_vars=y_var,
        height=5,
        aspect=0.6
    )
    g.map(sns.lineplot, ci=None, color='r', marker='o')

    for ax, x in zip(g.axes.ravel(), x_vars):
        ax.axhline(y=df[y_var[0]].mean(), ls='-.')
        ax.set_xlim([df[x].min(), df[x].max()])
        ax.set_xticks([df[x].min(), df[x].max()])

    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Main Effects Plot for Fitted Means', fontsize=16)
    g.axes[0][0].set_ylabel('Mean of y')
    g.axes[0][0].set_ylim([df[y_var[0]].min(), df[y_var[0]].max()])
  
    plt.show()

    return g.fig, g.axes[0]


def interact_plot(input_variables, y):
    '''...'''
    num_variables = len(input_variables)

    fig, axes = plt.subplots(
        num_variables,
        num_variables,
        figsize=(num_variables**2, num_variables**2),
        sharey=True
    )

    fig.suptitle('Interaction Plot for y', fontsize=16)

    xs = product(input_variables, input_variables)

    for i in range(num_variables):
        for j in range(num_variables):
            first_x, second_x = next(xs)

            if i != j:
                interaction_plot(
                    first_x, second_x, y,
                    ax=axes[j][i],
                    colors=['red','blue'],
                    markers=['D','^'],
                    ylabel='y',
                    xlabel=' '
                )
                axes[j][i].title.set_text(f'x{i + 1}*x{j + 1}')
            else:
                fig.delaxes(axes[i][j])

    plt.show()

    return fig, axes


def surf_plot(input_variables, y):
    '''...'''
    num_comb = int(comb(len(input_variables), 2))
    nrows = ceil(num_comb / 4)
    ncols = min(num_comb, 4)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharey=False,
        subplot_kw=dict(projection='3d'),
        figsize=(ncols * 4, nrows * 4)
    )
    fig.suptitle('Surface Plot for y', fontsize=16)

    xs = [
        np.linspace(x.min(), x.max(), 50)
        for x in input_variables
    ]

    for ax, ((i, x1), (j, x2)) in zip(axes.ravel(), combinations(enumerate(xs), 2)):
        x1_grid, x2_grid = np.meshgrid(x1, x2)
        iterpolated = scipy.interpolate.griddata(
            (input_variables[i], input_variables[j]),
            y,
            (x1_grid, x2_grid),
            method='cubic'
        )
        ax.plot_surface(x2_grid, x1_grid, iterpolated, cmap='viridis')
        ax.set_xlabel(f'\nx{j + 1}', fontsize=14)
        ax.set_ylabel(f'\nx{i + 1}', fontsize=14)
        ax.set_zlabel('\n y', fontsize=14)
        ax.set_zlim(y.min(), y.max())
        ax.view_init(10, 30)

    axes[0][0].view_init(30, 30) if num_comb > 4 else axes[0].view_init(30, 30)

    return fig, axes


def PSE(model, alpha):
    'Half Normal plot using Lenth’s method'
    results = model
    x = results.params.drop(labels='intercept')
    
    CD = _pd.DataFrame()
    CD['coef'] = x
    CD['effect'] = 2 * x
    CD['count'] = range(len(x))
    CD['osr'] = abs(2 * x)
    CD = CD.sort_values(by='osr')

    osm, osr = sps.probplot(CD['osr'], dist=sps.halfnorm, fit=False)
    CD['osm'] = osm

    x2 = CD['osr'][CD['osr'] <= 3.75 * CD['osr'].median()]
    PSE = 1.5 * x2.median()
    CD['T-value'] = CD['effect'] / PSE
    nu = int(len(x2) / 3)
    T = sps.t(nu)
    CD['p-value'] = np.around(2 - 2*T.cdf(abs(CD['T-value'])), 3)
    
    CD2 = CD[CD['p-value'] < alpha]
    CD = CD.sort_values(by='count')
    CD = CD.drop(labels=['count', 'osr', 'osm'], axis=1)
    
    print("Coded coefficients and Lenth's pseudo T- and p- values", CD, sep='\n\n')
    osm2 = [0, 2]
    osr2 = [0, 2 * PSE]
    plt.scatter(osm, osr)
    for term in CD2.index:
        plt.annotate(term, (CD2['osm'][term] + 0.05, CD2['osr'][term]))
        plt.plot(CD2['osm'][term], CD2['osr'][term], 'ro')
    
    plt.title(f"Daniels plot using Lenth's PSE with alpha = {alpha}")
    plt.xlabel('Half normal score')
    plt.ylabel('Absolute effect')
    plt.plot(osm2, osr2, 'r');

    return CD


def hist_bins(start, stop, step):
    '''...'''
    return np.arange(start, stop + step / 2, step).tolist()


def relative_SE(M: _pd.DataFrame):
    '''
    Estimates relative standard error of the generated design

    M is the dataframe containing all the model terms in the designed experiment
    that is including any interaction term
    '''
    x = M.values

    # Function to estimate Relative Standard Error
    kb = np.sqrt(np.diag(np.linalg.inv(np.dot(x.T, x))))
    
    return _pd.DataFrame(data=kb, index=M.columns, columns=['Relative SE'])


def leverage(M: _pd.DataFrame):
    '''
    Calculates leverage for the DoE

    `M` is the dataframe containing all the model terms in the designed experiment
    '''
    x = M.values
    a = np.linalg.inv(np.dot(x.T, x))
    b = np.dot(a, x.T)
    h = np.diagonal(np.dot(x, b))

    return _pd.DataFrame(data=h, columns=['Leverage'])


def orthogonality(M: _pd.DataFrame):
    '''
    Determines the Variance Inflation Factor (VIF) for the DoE

    `M` is the dataframe containing all the model terms in the designed experiment
    '''
    x = [
        variance_inflation_factor(M.values, i)
        for i in range(len(M.columns))
    ]
    vif = _pd.DataFrame(
        data=x,
        index=M.columns,
        columns=['VIF']
    ).drop(labels=['Intercept'], axis=0)
    
    return vif


def relative_PSD(input_variables, M):
    '''
    Determines the Relative Prediction Standard Deviation for the DoE

    `input_variables` are the linear terms of the design (x1, x2, ..., xn).
    One of the term needs to be held constant.

    `M` is the dataframe containing all the model terms in the designed experiment
    '''
    m = np.array(
        [1]
        + input_variables                                                           # x1, x2, ..., xn
        + [var_1 * var_2 for (var_1, var_2) in combinations(input_variables, 2)]    # x1*x2, x1*x3, ...
        + [var**2 for var in input_variables]                                       # x1^2, x2^2, ..., xn^2
    )
    A = np.linalg.inv(np.dot(M.T, M))       # (M' * M)^(-1)
    
    return m.dot(A.dot(m.T))


def printmd(x):
    display(Markdown(x))


class RegressionAnalysis():
    """Regression analysis class module

    Methods
    =======
    """
    def __init__(self, data, x, y, exclude=[], code=False):
        """Constructor

        Parameters
        ==========
        data: Pandas dataframe containing the response (y) and the
        regressors (x's).
        x: List of dataframe columns containing the x's.
        y: Str - the column containing the response.
        exclude: List (optional) of terms to exclude from the model
        code: Boolean (optional default=False) coded unit switch

        e.g.
        ra = RegressionAnalysis(data=df, x=['x1', 'x2', 'x3'], y='y',
                                exclude=['x1*x1', 'x2*x2', 'x3*x3'],
                                code=True)
        """
        self._data = data
        self._x = x
        self._y = y
        self._exclude = exclude
        self._code = code
        if self._code:
            printmd('## **Analysis being carried out using coded units**')
        self._model = self._fit_model() # ???
        self._n = len(self._endog)
        self.res = OLSInfluence(self._model)
        self.Y = _pd.DataFrame()
        self.Y['fits'] = self._model.fittedvalues
        self.Y['student_resid'] = self.res.summary_frame()['student_resid']
    
    @staticmethod
    def add_quadratic(df):
        # adds quadratic terms to the x's
        L = df.select_dtypes(include=[np.number]).columns.tolist()
        quad_col_pairs = list(combinations_with_replacement(L, 2))
        for col_pair in quad_col_pairs:
            col1, col2 = col_pair
            quadratic_col = f'{col1}*{col2}'
            df[quadratic_col] = df[col1] * df[col2]

    @staticmethod
    def _sign_str(x):
        if x < 0:
            return f' - {-x}'
        else:
            return f' + {x}'
    
    def _fit_model(self):
        return sm.OLS(endog=self._endog, exog=self._exog).fit()
    
    def summary(self):
        print(self._model.summary())
    
    def anova(self):
        data_dct = {
            'Source': ['Regression', 'Error', 'Total'],
            'DF': [
                int(self._model.df_model),
                int(self._model.df_resid),
                int(self._model.df_model + self._model.df_resid)
            ],
            'SS': [
                self._model.ess,
                self._model.ssr,
                self._model.centered_tss
            ],
            'MS': [
                self._model.mse_model,
                self._model.mse_resid,
                self._model.mse_total
            ],
            'F': [self._model.fvalue, np.nan, np.nan],
            'p': [self._model.f_pvalue, np.nan, np.nan]
        }

        out_df = _pd.DataFrame.from_dict(data_dct)
        
        return out_df.set_index('Source')\
            .style.format(
                {'DF': '{:6d}',
                 'SS': '{:20f}',
                 'MS': '{:10f}',
                 'F': '{:10.3f}',
                 'p': '{:10.3f}'},
                na_rep=''
            ).set_properties(**{'text-align': 'right', 'width': '80px'})

    def statistics(self):
        'Produces regression statistics'
        return self._model.summary()
    
    def _hdr_names(self):
        res = self.statistics()
        res_strs = str(res).split('\n')
        hdr_lst = res_strs[12].split()
        hdr_lst.remove('err')
        hdr_lst[1] = 'std err'
        return hdr_lst
    
    def _coef_names(self):
        res = self.statistics()
        res_strs = str(res).split('\n')
        return [
            res_strs[i].split()[0]
            for i in range(14, 14 + len(self._exog.columns))
        ]
    
    def _coef_table(self):
        res = self.statistics()
        res_strs = str(res).split('\n')
        return [
            [float(res_strs[j].split()[i]) 
             for i in range(1, 7)]
             for j in range(14, 14 + len(self._exog.columns))
        ]
    
    def rst_table(self):
        dfm = _pd.DataFrame(
            data=self._coef_table(),
            index=self._coef_names(),
            columns=self._hdr_names()
        )
        dfm['VIF'] = [
            variance_inflation_factor(self._exog.values, i)
            for i in range(len(self._exog.columns))
        ]
        return dfm
    
    def coeftable(self):
        dfm = _pd.DataFrame()
        dfm['coef'] = self._model.params
        dfm['std err'] = self._model.bse
        dfm['t'] = self._model.tvalues
        dfm['p-value'] = self._model.pvalues
        dfm['VIF'] = [
            variance_inflation_factor(self._exog.values, i)
            for i in range(len(self._exog.columns))
        ]
        return dfm
    
    def mdlquality(self):
        data = {
            'S': np.sqrt(self._model.mse_resid),
            'R2': [self._model.rsquared],
            'R2-adj': [self._model.rsquared_adj],
            'R2-pred': [1 - OLSInfluence(self._model).ess_press / self._model.centered_tss],
            'PRESS': [OLSInfluence(self._model).ess_press],
            'PRESS RMSE': [np.sqrt(OLSInfluence(self._model).ess_press/self._n)],
            'AIC': [self._model.aic],
            'BIC': [self._model.bic]
        }
        dfq = _pd.DataFrame(data)
        dfq.index = [' ']
        
        return dfq
    
    def _equation(self):
        cf = self.rst_table()['coef'].to_list()
        vr = self._exog.columns.to_list()
        out_str = f'{self._y} = cf[0]'
        #out_str += self._sign_str(cf[1]) + self._x
        for i in range(1, len(self._exog.columns)):
            out_str += self._sign_str(cf[i]) + vr[i]
        
        return out_str
    
    def equation(self):
        display(Latex(self._equation()))

    def results(self, round=3):
        """
        Table of results
        
        Parameters
        ==========
        round: int (optional) rounding (default is 3 decimal places)
        """
        printmd('## **Regression Analysis**')
        print('\n')
        self.equation()
        print('\n')
        display(self.anova())
        print('\n')
        display(self.rst_table().round(round))
        print('\n')
        display(self.mdlquality().round(round))

    def probplot(self, ax=None):
        '''
        Probability plot (using the Probscale Package)
        
        ax : pyplot axes object for plot
        '''
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        probscale.probplot(
            self.Y['student_resid'].dropna().to_numpy(),
            ax=ax,
            probax='y',
            bestfit=True
        )
        ax.set_title('Normal Probability Plot of the Deleted Residuals')
        ax.set_xlabel('Deleted Residuals')
    
    def resvfits(self, ax=None):
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        plt.scatter(self.Y['fits'], self.Y['student_resid'])
        ax = plt.gca()
        ax.set_title('Deleted Residuals vs Fitted Values')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Deleted Residuals')
    
    def reshist(self, ax=None):
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        
        plt.hist(
            self.Y['student_resid'], 
            bins=ceil(np.sqrt(self._n))
        )
        ax = plt.gca()
        ax.set_title('Histogram of Deleted Residuals')
    
    def resvorder(self, ax=None):
        if ax is None:
            ax = plt.gca()
        else:
            plt.sca(ax)
        x = range(1, 1 + self._n)
        plt.plot(x, self.Y['student_resid'], '-o')
        ax = plt.gca()
        ax.set_title('Deleted Residuals vs Order')
        ax.set_xlabel('Order')
        ax.set_ylabel('Deleted Residuals')

    def res4pack(self):
        fig, axs = plt.subplots(2, 2)
        fig.set_size_inches(12, 8)
        self.probplot(ax=axs[0, 0])
        self.resvfits(ax=axs[0, 1])
        self.reshist(ax=axs[1, 0])
        self.resvorder(ax=axs[1, 1])
        plt.tight_layout()

    @property
    def _exog(self):
        out = self._data[self._x].copy()
        if self._code:
            # Convert to coded units
            coder = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            d_arr = coder.fit_transform(out)
            out = _pd.DataFrame(d_arr, columns=out.columns)
        self.add_quadratic(out)
        out.insert(0, 'intercept', 1)
        out = out.drop(self._exclude, axis=1)

        return out
    
    @property
    def _endog(self):
        return self._data[self._y].copy()
    
    @property
    def regressors(self):
        return self._exog.index.to_list


class CCD():
    def __init__(
        self, k, ncp, alpha='face centered',
        fact_names=None, actual_lows=None, actual_highs=None
    ):
        assert [fact_names, actual_lows, actual_highs].count(None) in (0, 3), 'All three settings should be set up!'

        self.df = build_ccd(k, alpha, ncp)
        self.df.index = range(len(self.df))                                         # Reset the index column
        self.model = make_model(self.df.columns, ModelOrder.quadratic)              # Create quadratic model property
        self.mmatrix = dmatrix(self.model, data=self.df, return_type='dataframe')   # Generate model matrix
        self.dfr = self.df.copy()

        if [fact_names, actual_lows, actual_highs].count(None) == 0:
            self.dfr.columns = fact_names
            self.actual_lows = dict(zip(fact_names, actual_lows))
            self.actual_highs = dict(zip(fact_names, actual_highs))
            self.dfr = dexpy.design.coded_to_actual(self.dfr, self.actual_lows, self.actual_highs)
        
        self.response = _pd.Series(dtype=float)

    def fit(self):
        """
        Adds the response to the dataframes
        """
        self.df['Y'] = self.response
        self.dfr['Y'] = self.response


class DOpt():        # should it be combined with CCD()?
    def __init__(self, k, n, order=ModelOrder.quadratic):
        self.df = build_optimal(k, run_count=n, order=order)
        self.model = make_model(self.df.columns, ModelOrder.quadratic)              # Create quadratic model property
        self.mmatrix = dmatrix(self.model, data=self.df, return_type='dataframe')   # Generate model matrix


class pd:
    """
    This is a class of static methods that are intended as utilities that extend
    pandas, e.g.

    import s4eu as su
    series = su.pd.Series.fromSample(sample)
    """

    @staticmethod
    def sigfigs(
        df: _Union[_pd.DataFrame, _pd.Series], 
        nfigs: int, 
        inplace: bool = False
    ) -> _Union[_pd.DataFrame, _pd.Series, None]:
        """
        This function rounds a Series/DataFrame to a specified number of significant
        figures (not decimal places) for tidy display, e.g.

        display(s4eu.pd.sigfigs(df), 3)
        """
        
        # Create a positive data vector with a place holder for NaN / inf data
        data = df.values
        data_positive = _np.where(
            _np.isfinite(data) & (data != 0),
            _np.abs(data),
            10**(nfigs-1))
        # Align data by magnitude, round, and scale back to original
        magnitude = 10 ** (nfigs - 1 - _np.floor(_np.log10(data_positive)))
        data_rounded = _np.round(data * magnitude) / magnitude
        # Place back into Series or DataFrame
        if inplace:
            df.loc[:] = data_rounded
        else:
            if isinstance(df, _pd.DataFrame):
                return _pd.DataFrame(
                    data=data_rounded,
                    index=df.index,
                    columns=df.columns)
            else:
                return _pd.Series(data=data_rounded, index=df.index)
