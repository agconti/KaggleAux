import sys
from datetime import timedelta
from pandas.io.data import DataReader


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
    def __init__(self, total_tasks):
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
                      data_source='yahoo').ix[0, 5]


def quater_maker(d):
    '''
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

    '''
    d = d.month
    quaters = {
        "Q1": [1, 2, 3],
        "Q2": [4, 5, 6],
        "Q3": [7, 8, 9],
        "Q4": [10, 11, 12]
    }
    return list((key for key in quaters.keys() if d in quaters[key]))[0]
    