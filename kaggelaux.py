# Kaggel Auxillary Functions
# AGC 2013

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
    import sys

    progress = "\r["
    
    for _ in range (0,i):
        progress += "#"
    
    for _ in range (i, num_tasks):
        progress += " "
    
    progress += "]"
    
    sys.stdout.write(progress)
    sys.stdout.flush()
    

def regress_pred_output(test_data, z, y = ""):
    """ 
    Returns a dataframe of independ var predictions of a test file based on your regression of a train file.
    
    Parameters
    --
    Test_data: must be the test data sheet opened through pandas: ie. in a dataframe 
    z: should be results object from training regeresion from the SciPy statsmodels resuts object
    y: should be the dependent var in a string from your traing data, if left blank; the function will try to parse it for you.(will only work w/o spaces in the variable name) 
    --
   
    Returns
    --
    Returns a merged dataframe
    AGC 2013
    """
    import numpy as np
    from pandas import Series
    import pylab
    from patsy import dmatrices
    from re import split
    model_dep=''
    y_holder=0
    
    ######## House Keeping #######
    #parse independt var
    if y == '':
        s = z.summary_old()
        s = split('[\s\n|]',s)
        y = s[26]
    #make sure y is in data frame
    if (y in test_data.columns) == False:
        indep_header = str(y) + " test constant for calculations"
        test_data[indep_header] = np.random.randn()

    #convert results to series
    z = Series(z.params)
    
    
    #arrange our independ vars
    for i in z.index:
        if i != 'Intercept':
            if '[' in i:
                #get around catagorical splits
                model_holder = split("[[]", i)
                i = model_holder[0]
            #check for duplicates
            if (i in model_dep) == False:
                #deal with first element added
                if model_dep == '':
                    model_dep = str(i)
                else:
                  #create string of indep vars
                    model_dep = model_dep + ' + ' + str(i)
          
    
    
    #put our model togehter
    model = (y + ' ~ ' + model_dep)
    print model
    #create reg friendly exog and endog dfs
    df_I, df_D = dmatrices( model, data=test_data, return_type='dataframe')
    
    #create predictions
    for x in xrange(0, len(df_I)):
        #sum of B*x
        ##ex: 'SalePrice ~ YearMade + MachineID + ModelID + datasource + auctioneerID + UsageBand'
        y_holder = 0
        for i in z.index:
            y_holder += (df_D.ix[x, str(i)] * z[str(i)])
        test_data[y].ix[x] = y_holder
        
    #reg_results=pandas.merge(df_I,df_D, left_index=True,right_index=True)
    #results=pandas.concat(df_I,test_data, left_index=True,right_index=True)
    return test_data

def quater_maker(d):
    """
    Pareses dates and returns the apropriate quater.
    --
    Parameters
    d: a python date time object
    --
    Returns:
    The quater in a string
    
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
    df2 = the soultions set
    p = option to print rmsle as string; 0 = dont print, 1 = print 
    outs
    --
    prints rmsle
    rmsle as a float
    '''
    import numpy as np
    
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsmle = np.sqrt(np.mean(np.power(np.log(actual + 1) - np.log(prediction + 1), 2)))
    
    if p == 1 :
        print "rsmle: " + str(rsmle)
    return rsmle

def score_rmse(y, df, df2, p = 0 ):
    """
    ins 
    --
    y =  what your trying to predict. must be a string. ie. 'SalesPrice'
    df =  your predictions
    df2 = the soultions set
    p = option to print rmsle as string; 0 = dont print, 1 = print

    outs
    --
    prints rmse
    rmse as a float
    """
    import numpy as np

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
    s = signifcance level, default at 90% confidence
    
    Outs
    --
    returns a list of columns below siginicance level
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
    from pandas.io.data import DataReader
    from datetime import date, timedelta
 
    x = (x - timedelta(days = lag))
    r = DataReader(ticker,start=x, end=x, data_source='yahoo')
    r = r.ix[0,5]
    return r

def side_by_side(*objs, **kwds):
    from pandas.core.common import adjoin
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
    from pandas import Series, DataFrame
    sum_stats = []
    for i in df.columns:
        x = Series(df[i].describe())
        x.name = i
        sum_stats.append(x)
    stats = DataFrame(sum_stats)
    stats
    return stats
