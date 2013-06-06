##AGC_KaggleAux
A module of auxiliary functions to aid competitors in Kaggle competitions. You may use these functions only in accordance with the included license.


Functions include:
* def ml_formula(y, df):
    '''
    a simple function to create a formula using all available features for patsy dmatrices function. 

    ins  
    --
    y = a string of the variable your trying to predict

    df = the data frame your trying to make your predictions on 

    outs 
    ---
    a string of the desired formula
    '''

* def progress(i, num_tasks):
    '''
    A simple textual progress bar
    
    Ins
    --
    i = should be an iterable value to measure the progress of your task
    num_tasks = total number of tasks

    outs 
    --
    A progress bar like [#########  ]
    '''
    
* def cat_clean(s):
    """
    Cleans categorical variables of NaNs by setting the NaN to 0 so it can be processed by a patsy dmatrices function.

    Inputs
    --
    s = cat value

    output
    --
    cat value or int 0 in place of a NaN val
    """

def predict(test_data, results, i):
    """ 
    Returns a NumPy array of independent variable predictions of a test file based on your regression of a train file. Built for speed
    
    Parameters
    --
    Test_data: should be test data you are trying to predict in a pandas dataframe 
    results: should be dict of your models results wrapper and the formula used to produce it. 
        ie.  
        results['Model_Name'] = {[<statsmodels.regression.linear_model.RegressionResultsWrapper> , "Price ~ I(Supply, Demand)] }
    i: should be the name of your model. You can iterate through the results dict. 
    --
   
    Returns
    --
    Predictions in a flat NumPy array. 
    AGC 2013
    """

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

def score_rmsle(y, df, df2, p = 0):
    '''
    ins 
    --
    y =  what your trying to predict. must be a string. ie. 'SalesPrice'
    df =  your predictions
    df2 = the solutions set
    p = option to print rmsle as string; 0 = don't print, 1 = print 
    outs
    --
    prints rmsle
    rmsle as a float
    '''

def score_rmse(y, df, df2, p = 0 ):
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
