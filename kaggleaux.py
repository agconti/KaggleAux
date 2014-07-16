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
    Return predictions of based on model results.

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


def category_boolean_maker(series):
    '''
    A function for to designate missing records from observed ones.

    When used with the pandas df.series.apply() method, it will create
    a boolean category variable. If values exist the bool will register
    as 1, if nan values exist the bool will register as 0.

    Parameters
    ----------
    series : Series
        A pandas series to perform comparison on.

    Returns
    -------
    Int :
        0 or 1 for missing values.

    '''
    return 0 if np.isnan(series) is True else 1


def columns_to_str(column_list, operand=', ', return_list=False):
    '''
    Return the list of features as strings for easy implementation patsy formulas.

    Parameters
    ----------
    column_list : list
        A list of features, usually from generated from pandas's df.columns function.

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
        the data frame your trying to make your predictions on.

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
        The corresponding quarter in a string.
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
    float :
        rmsle
    '''
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsmle = np.sqrt(np.mean(np.power(np.log(actual + 1) - np.log(prediction + 1), 2)))

    if p:
        print "rsmle: {0}".format(rsmle)
    return rsmle


def score_rmse(y, df, df2, p=False):
    """
    Returns the Root Mean Squared Error of predictions.

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
        rmse
    """
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsme = np.sqrt(np.mean(np.power((actual - prediction), 2)))

    if p:
        print "rsme: {0}".format(rsme)

    return rsme


def filter_features(model_results, significance=0.1):
    '''
    Returns a list of features that are below a given level of significance.

    Parameters
    ----------
    model_results : Series
        a pandas series of the results.pvalues of your model
    significance : float
        significance level, default at 90% confidence.

    Returns
    -------
    list :
         a list of columns below the given significance level
    '''
    return list((model_results.index[index] for index, pvalues in enumerate(model_results)
                if pvalues > significance))


def stock_price_at_date(lookup_date, ticker, lag=0):
    '''
    Returns the daily share price of a stock for a specified date range.

    Parameters
    ----------
    lookup_date : Datetime.datetime.date
         End date.
    ticker: str
        str corresponding the the stock's ticker symbol.
    lag : int
        length of trading days before start date

    Returns
    --------
    DataFrame :
        A stock's prices over a the given period.
    '''
    start = (lookup_date - timedelta(days=lag))
    return DataReader(ticker, start=start, end=lookup_date,
                      data_source='yahoo').ix[0,5]


def describe_frame(df):
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
