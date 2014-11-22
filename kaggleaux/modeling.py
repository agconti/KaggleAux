import numpy as np
from pandas import DataFrame
from patsy import dmatrices
import dataframe as ka_df


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
            results['Model_Name'] = [<statsmodels.regression.linear_model.RegressionResultsWrapper>,
                                     "Price ~ I(Supply, Demand)]
    model_name: str
        should be the name of your model. You can iterate through the results dict.

    Returns
    -------
    NumPy array
        Predictions in a flat NumPy array.

    Example
    -------
    results = {
        'Logit': [<statsmodels.discrete.discrete_model.BinaryResultsWrapper at 0x117896650>,
                  'survived ~ C(pclass) + C(sex) + age + sibsp  + C(embarked)']
    }
    compared_resuts = predict(test_data, results, 'Logit')

    """
    model_params = DataFrame(results[model_name][0].params)
    formula = results[model_name][1]

    # Create regression friendly test DataFrame
    yt, xt = dmatrices(formula, data=test_data, return_type='dataframe')
    xt, model_params = ka_df.get_dataframes_intersections(xt, xt.columns,
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
