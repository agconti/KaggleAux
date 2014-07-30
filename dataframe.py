## Auxiliary functions for Pandas DataFrames ##
import numpy as np
import scipy as sp
from pandas import DataFrame, Series


def describe(df):
    """
    Returns the descriptive statistics of a given dataframe.

    Parameters
    ----------
    df : DataFrane
        the dataframe you want to describe.

    Returns
    -------
    DataFrame :
        descriptive stats on your dataframe, in dataframe.

    """
    sum_stats = []
    for i in df.columns:
        x = Series(df[i].describe())
        x.name = i
        sum_stats.append(x)
    stats = DataFrame(sum_stats)
    return stats


def get_intersection(df, comparator1, comparator2):
    """
    Return a dataframe with only the columns found in a comparative dataframe.

    Parameters
    ----------
    comparator1: DataFrame
        DataFrame to preform comparison on.
    comparator2: DataFrame
        DataFrame to compare with.

    Returns
    -------
    DataFrame:
        Data frame with columns not found in comparator dropped.

    """
    to_drop = list((c for c in comparator1 if c not in comparator2))
    return df.drop(to_drop, axis=1)


def get_dataframes_intersections(df1, comparator1, df2, comparator2):
    """
    Return DataFrames with the intersection of their column values.

    Parameters
    ----------
    comparator1: DataFrame
        DataFrame to preform comparison on.
    comparator2: DataFrame
        DataFrame to compare with.

    Returns
    -------
    Tuple:
        The resultingDataframe with columns not found in comparator dropped.

    """
    comparator1 = get_intersection(df1, comparator1, comparator2)
    comparator2 = get_intersection(df2, comparator2, comparator1)
    return comparator1, comparator2


def cross_validate_df(df, percent):
    '''
    Return a randomly shuffled subsets of a DataFrame cross validation
    or for down sampling.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be sampled. Expects an ordinal index; ie. 0 - 100.
    percent: Int
         the percentage to split of the returned subsets.

    Returns
    -------
    Tuple:
        (df_h1, df_h2), both parts of the split randomly shuffled DataFrame

    Example
    -------
    small_df_half, large_df_half  = cross_validate_df(df, 33)

    '''
    sample_percentage_of_dataframe = int(np.round(((percent * df.index.size) / float(100))))
    rows = np.random.randint(0, df.index.size, sample_percentage_of_dataframe)
    return (df.ix[rows], df.drop(rows))


def kfolds(df):
    '''
    Standard kfolds cross cross_validate method.

    Parameters
    ----------
    df : DataFrame
        A pandas dataframe to be operated on.

    Returns
    -------
    Tuple :
        dataframe 10%, dataframe 90% -- random splits using python's random.choice()

    '''
    return cross_validate_df(df, 90)


def welch_ttest(df, described_frame, boolean_feature):
    '''
    Parameters
    ----------
    df : DataFrame
       A DataFrame to perform welch_ttest on.
    described_frame : DataFrame
       A described DataFrame from the pandas desribe_frame() method.
    boolean_feature: Str
       Name of boolean feature to conduct test on.

    Returns
    -------
    DataFrame :
        t-statistic and p-value for each feature in a pandas dataframe.

    '''

    described_frame['t-statistic'] = np.nan
    described_frame['p-value'] = np.nan

    for name, item in df.iteritems():
        result = sp.stats.ttest_ind(df[name][df[boolean_feature] == 0].dropna(),
                                    df[name][df[boolean_feature] == 1].dropna(),
                                    equal_var=False)
        described_frame.ix[name, 't-statistic'], described_frame.ix[name, 'p-value'] = result
    return described_frame
    