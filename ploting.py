import numpy as np
import matplotlib as plt
from pandas import DataFrame, qcut


def bin_residuals(residuals, feature, bin_count):
    '''
    Returns the average binned residuals of a feature.

    Returns a dataframe indexed by the bins, with the bin midpoint,
    the residual average within the bin, and the confidence interval
    bounds.

    Parameters
    ----------
    residuals :
        The residuals of the predictions of a feature from a particular model.
    
    feature : Series or ndarray
        A feature and it's observations to average
    
    bin_count : int
        The number of bins to use for averaging the residuals.
        ie. bin_count = 4 ; # makes quartiles.

    Returns
    -------
    DataFrame :
        A DataFrame containing the average binned result of a feature.

    '''
    residuals_df = DataFrame({'feature': feature, 'residuals': residuals})
    
    residuals_df['bin_count'] = qcut(feature, bin_count)
    bin_group = residuals_df.groupby('bin_count')
    
    bin_df = bin_group['feature', 'residuals'].mean()
    bin_df['count'] = bin_group['residuals'].count()
    bin_df['lower_ci'] = (-2 * (bin_group['residuals'].std() /
                                np.sqrt(bin_group['residuals'].count())))
    bin_df['upper_ci'] = (2 * (bin_group['residuals'].std() /
                               np.sqrt(bin_df['count'])))
    bin_df = bin_df.sort('feature')
    return(bin_df)


def plot_binned_residuals(bin_df):
    '''
    Plots the binned residual averages and confidence intervals of a binned dataframe.

    Parameters
    ----------
    bin_df : DataFrame
       the binned dataframe from bin_residuals(residuals, feature, bin_count).
    
    Returns
    -------
    matplotlib.figure :
        Plot of data frame residuals and confidence intervals.
    '''
    plt.plot(bin_df['var'], bin_df['resid'], '.')
    plt.plot(bin_df['var'], bin_df['lower_ci'], '-r')
    plt.plot(bin_df['var'], bin_df['upper_ci'], '-r')
    plt.axhline(0, color='gray', lw=0.5)
    return plt
