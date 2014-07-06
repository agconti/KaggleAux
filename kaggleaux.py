# Kaggel Auxillary Functions
# AGC - 2013
import sys
from datetime import timedelta
import numpy as np
import scipy as sp
import matplotlib as plt
from pandas import DataFrame, Series, qcut
from pandas.core.common import adjoin
from pandas.io.data import DataReader
from patsy import dmatrices


def get_dataframe_intersection(df, comparator1,comparator2):
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
    comparator1 = get_dataframe_intersection(df1, comparator1, comparator2)
    comparator2 = get_dataframe_intersection(df2, comparator2, comparator1)
    return comparator1, comparator2


def predict(test_data, results, model_name):
    """
    Return predictions of based on model resutls.

    Parameters
    ----------
    test_data: DataFrame
        should be test data you are trying to predict
    results: Dict
        should be dict of your models results wrapper and the formula used
        to produce it.
            ie.
            results['Model_Name'] = {[<statsmodels.regression.linear_model.RegressionResultsWrapper> , "Price ~ I(Supply, Demand)] }
    model_name: Str
        should be the name of your model. You can iterate through the results dict.

    Returns
    -------
    NumPy array
        Predictions in a flat NumPy array.

    Example
    -------
    results = {'Logit': [<statsmodels.discrete.discrete_model.BinaryResultsWrapper at 0x117896650>,
               'survived ~ C(pclass) + C(sex) + age + sibsp  + C(embarked)']}
    compared_resuts = predict(test_data, results, 'Logit')

    """
    model_params = DataFrame(results[model_name][0].params)
    formula = results[model_name][1]

    # Create regression friendly test DataFrame
    yt, xt = dmatrices(formula, data=test_data, return_type='dataframe')
    xt, model_params = get_dataframes_intersections(xt, xt.columns,
                                                    model_params, model_params.index)
    # Convert to NumPy arrays for performance
    model_params = np.asarray(model_params)
    yt = np.asarray(yt)
    yt = yt.ravel()
    xt = np.asarray(xt)

    # Use our models to create predictions
    row, col = xt.shape
    model_parameters = model_params.ravel()
    model_array = list((model_parameters for parameter in xrange(row)))
    model_array = np.asarray(model_array)

    # Multiply matrix together
    predictions = np.multiply(xt, model_array)
    predictions = np.sum(predictions, axis=1)
    return predictions


def cross_validate_df(df, percent):
    '''
    Return a randomly shuffled supbsets of a DataFrame cross validation
    or for down sampleing.

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


def dataframe_kfolds(df):
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


def dataframe_welch_ttest(df, described_frame, boolean_feature):
    '''
    Parameters
    ----------
    df : DataFrame
       A DataFrame to perfrom welch_ttest on.
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


def category_boolean_maker(series):
    '''
    A funtction for to designate missing records from observed ones.

    When used with the pandas df.series.apply() method, it will create
    a boolean category variable. If values exist the bool will register
    as 1, if nan values exist the bool will register as 0.

    Parameters
    ----------
    series : Series
        A pandas series to perform comparision on.

    Returns
    -------
    Int :
        0 or 1 for missing values.

    '''
    return 0 if np.isnan(series) == True else 1


def columns_to_str(column_list, operand=', ', return_list=False):
    '''
    Return the list of features as strings for easy implementaiton patsy formulas.

    Parameters
    ----------
    column_list : list
        A list of features, ussually from generated from pandas's df.columns function.

    operand : str
        a sting to join list together by. Default is a comma: ', '. Could be a plus: ' + '
        for patsy equations.

    return_list : boolean
        ( optional ) return the list of features typecasted to str.

    Returns
    -------
    list :
        a list with elements typecasted to str.

    Example
    -------

    df.columns
    >>> Index([x3yv_E, x3yv_D, x1yv_E, x1yv_D], dtype=object)

    columns_to_str(df.columns)
    >>> Index(['x3yv_E', 'x3yv_D', 'x1yv_E', 'x1yv_D'], dtype=object)
    >>> ['x3yv_E', 'x3yv_D', 'x1yv_E', 'x1yv_D']

    '''
    if return_list == True:
        return list((str(feature) for feature in column_list))

    print "[%s]" % operand.join(map(str, column_list))


def add_to_model_subspace(left, right):
    '''
    Returns the intersection between two lists.

    Useful for defining a model's subspace variables with new added variables.

    Parameters
    ----------
    left : list
        a list you'd like to add to.

    right : list
        a list of things your trying to add

    Returns
    -------
    list :
        a list of the intersection between the passed in left and right lists.

    '''
    return list((str(right[i]) for i in xrange(0, len(right))
                if right[i] in left == False ))

def ml_formula(y, df):
    '''
    a simple function to create a formula using all available features for patsy dmatrices function.

    ins
    --
    y = a string of the variable your trying to predict

    df = the data frame your trying to make your predctions on

    outs
    ---
    a string of the desired formula
    '''
    formula = y +' ~ '
    for i, val in enumerate(df.columns):
        if i == 0 and val != y:
            formula += val
        if i != 0 and val != y:
            formula += ' + ' + val
    return formula



def progress(i, num_tasks):
    '''
    A simple textual progress bar

    Ins
    --
    i = should be an iterable value to measure the progress of your taks
    num_tasks = total number of tasks

    outs
    --
    A progress bar like [#########  ]
    '''
    progress = "\r["

    for _ in range (0,i):
        progress += "#"

    for _ in range (i, num_tasks):
        progress += " "

    progress += "]"

    sys.stdout.write(progress)
    sys.stdout.flush()

def cat_clean(s):
    """
    Cleans categorical variables of NaNs by setting the NaN to 0 so it can be processed by a patsy dmatrices function.

    Inputs
    --
    s = cat value

    output
    --
    cat value or int 0 in place of a NaN val
    """
    if isinstance(s, str) == False and np.isnan(s) == True:
        s = 0
    return s


def quater_maker(d):
    """
    Pareses dates and returns the appropriate quarter.
    --
    Parameters
    d: a python date time object
    --
    Returns:
    The quarter in a string

    AGC 2013
    """
    r = ''
    Q1 = [1, 2, 3]
    Q2 = [4, 5, 6]
    Q3 = [7, 8, 9]
    Q4 = [10, 11, 12]
    if d.month in Q1:
        r = 'Q1'
    if d.month in Q2:
        r = 'Q2'
    if d.month in Q3:
        r = 'Q3'
    if d.month in Q4:
        r = 'Q4'
    return r

      #if date is within daterange, Q1 ect. then output ex: 2011 Q1
      #manipulate months. say if  0<m4: return YYYY + Q1 ect.
      # then apply to dateline

def score_rmsle(y, df, df2, p = 0):
    '''
    ins
    --
    y =  what your trying to predict. must be a string. ie. 'SalesPrice'
    df =  your predictions
    df2 = the solutions set
    p = option to print rmsle as string; 0 = dont print, 1 = print
    outs
    --
    prints rmsle
    rmsle as a float
    '''
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsmle = np.sqrt(np.mean(np.power(np.log(actual + 1) - np.log(prediction + 1), 2)))

    if p == 1 :
        print "rsmle: " + str(rsmle)
    return rsmle

def score_rmse(y, df, df2, p=0 ):
    """
    ins
    --
    y =  what your trying to predict. must be a string. ie. 'SalesPrice'
    df =  your predictions
    df2 = the solutions set
    p = option to print rmsle as string; 0 = don't print, 1 = print

    outs
    --
    prints rmse
    rmse as a float
    """
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsme = np.sqrt(np.mean(np.power((actual - prediction), 2)))

    if p == 1:
        print "rsme: " + str(rsme)

    return rsme

def unwanted_pals(x, s = .1):
    '''
    Inputs
    --
    x should be a pandas series of the results.pvalues of your model
    s = significance level, default at 90% confidence

    Outs
    --
    returns a list of columns below significance level
    '''
    dropl = list()
    for i,z in enumerate(x):
        if z>s:
             dropl.append(x.index[i])
    return dropl

def stock_price_at_date(x, ticker, lag=0):
    '''
    ins
    --
    x should be the date your looking for
    ticker should be the stock ticker
    lag should be # of days to lag stock price

    outs
    --
    stock price.
    '''
    x = (x - timedelta(days = lag))
    r = DataReader(ticker,start=x, end=x, data_source='yahoo')
    r = r.ix[0,5]
    return r

def side_by_side(*objs, **kwds):
    '''
    created by wes mickinney, it only exists here becuase I use this function all the time.
    '''
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print adjoin(space, *reprs)

def describe_frame(df):
    """
    ins
    ---
    df = dataframe you want to describe

    outs
    ---
    descriptive stats on your dataframe, in dataframe.

    agc2013
    """
    sum_stats = []
    for i in df.columns:
        x = Series(df[i].describe())
        x.name = i
        sum_stats.append(x)
    stats = DataFrame(sum_stats)
    stats
    return stats

def bin_residuals(resid, var, bins):
    '''
    Compute average residuals within bins of a variable.

    Returns a dataframe indexed by the bins, with the bin midpoint,
    the residual average within the bin, and the confidence interval
    bounds.

    ins
    --
    resid, var, bins

    out
    --
    bin DataFrame

    '''
    resid_df = DataFrame({'var': var, 'resid': resid})
    resid_df['bins'] = qcut(var, bins)
    bin_group = resid_df.groupby('bins')
    bin_df = bin_group['var', 'resid'].mean()
    bin_df['count'] = bin_group['resid'].count()
    bin_df['lower_ci'] = -2 * (bin_group['resid'].std() /
                               np.sqrt(bin_group['resid'].count()))
    bin_df['upper_ci'] =  2 * (bin_group['resid'].std() /
                               np.sqrt(bin_df['count']))
    bin_df = bin_df.sort('var')
    return(bin_df)

def plot_binned_residuals(bin_df):
    '''
    Plotted binned residual averages and confidence intervals.

    ins
    --
    bin_df ie from bin_residuals(resid, var, bins)
    outs
    --
    pretty plots
    '''
    plt.plot(bin_df['var'], bin_df['resid'], '.')
    plt.plot(bin_df['var'], bin_df['lower_ci'], '-r')
    plt.plot(bin_df['var'], bin_df['upper_ci'], '-r')
    plt.axhline(0, color = 'gray', lw = .5)
