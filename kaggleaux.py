# Kaggel Auxillary Functions
# AGC - 2013
import sys
from datetime import timedelta
import numpy as np
import scipy as sp
import matplotlib as plt
from pandas import DataFrame, Series, qcut
from pandas.io.data import DataReader
from patsy import dmatrices


def get_dataframe_intersection(df, comparator1, comparator2):
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
    results: dict
        should be dict of your models results wrapper and the formula used
        to produce it.
            ie.
            results['Model_Name'] = {[<statsmodels.regression.linear_model.RegressionResultsWrapper> , "Price ~ I(Supply, Demand)] }
    model_name: str
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
    return 0 if np.isnan(series) is True else 1


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
    if return_list:
        return list((str(feature) for feature in column_list))

    print "[%s]" % operand.join(map(str, column_list))


def list_intersection(left, right):
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
                if right[i] in left is False))


def ml_formula(y, df):
    '''
    Returns a string as a formula for patsy's dmatrices function.

    Parameters
    ----------
    y : str
        a string of the variable your trying to predict.
    df : DataFrame
        the data frame your trying to make your predctions on.

    Returns
    -------
    str :
        a string of the desired formula

    Example
    -------
    y = "icecream_sales"
    df = pandas.DataFrame(columns=['icecream_sales','sunny_days', 'incidence_of_lactose_intolerance'])

    ml_formula(y, df)
    >>> 'icecream_sales ~ sunny_days + incidence_of_lactose_intolerance'
    '''
    independent_variable = y + ' ~ '

    dependent_variables = ' + '.join((val for i, val in enumerate(df.columns) if val != y))
    formula = independent_variable + dependent_variables
    return formula


class ProgressBar(object):
    '''
    Return A simple textual progress bar.

    Parameters
    ----------
    tasks_completed : int
        should be an incremented value to measure the progress of your tasks.
    total_tasks : int
        total number of tasks

    Retruns
    -------
    str :
        A progress bar like [#########  ]

    Example
    -------
    progress_bar = ProgressBar(2)
    def do_stuff():
        # do somethings

    progress_bar.update()
    >>>[# ]

    def do_more_things():
        # do so many things

    progress_bar.update()
    >>>[##]

    '''
    def __init__(total_tasks):
        self.total_tasks = total_tasks
        self.tasks_completed = 0
        self.progress = "\r["

    def increment(self):
        self.tasks_completed += 1

        for completed_task in xrange(0, self.tasks_completed):
            self.progress += "#"

        for incomplete_task in xrange(self.tasks_completed, self.total_tasks):
            self.progress += " "

    def show(self):
        sys.stdout.write(self.progress + "]")
        sys.stdout.flush()

    def update(self):
        self.increment()
        self.show()


def category_clean(category_value):
    """
    Retun a 0 for np.NaN values.

    Useful for cleaning categorical variables in a df.apply() function by
    setting the NaN to 0 so it can be processed by a patsy dmatrices function.

    Parameters
    ----------
    category_value :
        the category_value to be processed.

    Retruns
    -------
        str / category_value :
            category_value or int(0) in place of a NaN value.

    """
    if isinstance(category_value, str) is False and np.isnan(category_value) is True:
        return 0
    return category_value


def quater_maker(d):
    """
    Return the corresponding quarter for a given date.

    Parameters
    ----------
    d: datetime object
        The date you'd like to process

    Returns
    -------
    str :
        The coresponding quarter in a string.
        ie. 'Q1', 'Q2', 'Q3',  or 'Q4'

    """
    quaters = {
        "Q1": [1, 2, 3]
        "Q2": [4, 5, 6]
        "Q3": [7, 8, 9]
        "Q4": [10, 11, 12]
    }
    return key if d in quaters[key] for key in quaters.keys()


def score_rmsle(y, df, df2, p=False):
    '''
    Returns the Root Mean Squared Logarithmic Error of predictions

    Parameters
    ----------
    y : str
        Dependent Variable.
    df : DataFrame
        The predictions set.
    df2 : DataFrame
        The solutions set.
    p : bool
        option to print rmsle as string.

    Returns
    -------
    Float :
        rmsle
    '''
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsmle = np.sqrt(np.mean(np.power(np.log(actual + 1) - np.log(prediction + 1), 2)))

    if p:
        print "rsmle: {0}".format(rsmle)
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


def describe_frame(df):
    """
    ins
    ---
    df = dataframe you want to describe

    outs
    ---
    descriptive stats on your dataframe, in dataframe.

    """
    sum_stats = []
    for i in df.columns:
        x = Series(df[i].describe())
        x.name = i
        sum_stats.append(x)
    stats = DataFrame(sum_stats)
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
