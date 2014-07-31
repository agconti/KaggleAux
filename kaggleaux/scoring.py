import numpy as np


def rmsle(y, df, df2, p=False):
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


def rmse(y, df, df2, p=False):
    '''
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
    '''
    prediction = np.asarray(df[y])
    actual = np.asarray(df2[y])

    rsme = np.sqrt(np.mean(np.power((actual - prediction), 2)))

    if p:
        print "rsme: {0}".format(rsme)

    return rsme
