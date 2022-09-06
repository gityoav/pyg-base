from pyg_base._dictable import dictable
from pyg_base._as_list import as_list
from pyg_base._dates import dt
from pyg_base._pandas import df_slice
from pyg_base._dictattr import dictattr
import pandas as pd

_data = 'data'

def _min(*values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None
    else:
        return min(values)

def _max(*values):
    values = [v for v in values if v is not None]
    if len(values) == 0:
        return None
    else:
        return max(values)

def df_roll_off(chain, loader,load_on = None,transform = None,
            roll = 'roll', expiry = '-1m', n = 1, data = None, cutoff = '-1b'):
    """
    
    Creates a single timeseries from multiple timeseries data, 
    We assume we roll off a timeseries when it expires

    Parameters
    ----------
    chain : dictable
        list of timeseries to be loaded
    loader : callable
        function to load timeseries
    load_on : str/list of str
        columns to use when loading. The default is None.
    transform : callable, optional
        a transform to apply to the data. The default is None.
    roll: str, optional
        the column used to determine roll. The default is 'roll'.
    expiry : str/int, optional
        Relative date to today that determines the expiry of available timeseries, The default is '-1b'.
    n : integer, optional
        The size of the resulting curve. The default is 1.
    data : timeseries, optional
        The rolled timeseries, previously calculated . The default is None.
    cutoff : str/int, optional
        old data will be used up to that relative date

    Returns
    -------
    timeseries. A rolled timeseries
    
    Example: roll when each time series ends
    --------
    >>> from pyg import * 
    >>> def loader(end_date, value):
    >>>     return pd.Series(value, drange(-800, min(dt(0), dt(end_date))))

    >>> end_dates = drange(dt(2000,4,0), dt(1000), '3m')
    >>> chain = dictable(value = range(len(end_dates)), end_date = end_dates)
    >>> load_on = None; transform = None; roll = 'roll'; expiry = '-1m'; n = 1; data = None; cutoff = '-1b'

    >>> d = df_roll_off(chain, loader = loader)

    >>> d.chain ## we roll when data ends
    dictable[51 x 3]
    value|end_date           |roll               
    0    |2000-03-31 00:00:00|2000-03-31 00:00:00
    1    |2000-12-31 00:00:00|2000-12-31 00:00:00
    2    |2001-03-31 00:00:00|2001-03-31 00:00:00
    ...51 rows...
    48   |2024-03-31 00:00:00|None               
    49   |2024-12-31 00:00:00|None               
    50   |2025-03-31 00:00:00|None               

    >>> d.data[diff(d.data) > 0]
    2000-04-01     1
    2001-01-01     2
    ...
    2022-01-01    44
    2022-04-01    45
    
    
    Example: extend an existing result
    -----------------------------------
    >>> d2 = df_roll_off(chain, loader = loader, data = d.data.iloc[-100:])
    >>> assert eq(pd.DataFrame(d.data), pd.DataFrame(d2.data))
    
    Example: roll dates specified
    -----------------------------
    >>> chain = chain(roll = lambda end_date: dt(end_date, -15)) - 'data' ## roll mid-month
    >>> d3 = df_roll_off(chain, loader = loader).data
    >>> d3[diff(d3)>0]
    Out[163]: 
    2000-03-17     1 # <--- mid month rolls
    2000-12-17     2
                  ..
    2021-12-17    44
    2022-03-17    45
    Length: 45, dtype: int64
    

    >>> d3 = df_roll_off(chain, loader = loader, data = d2.data[:dt(2010)]).data 
    >>> d3[diff(d3)>0]
    Out[165]: 
    2000-04-01     1 ## <--- historic data that rolled end-of-month
    2001-01-01     2
                  ..
    2021-12-17    44 ## <--- data from 2010 that rolls mid-month
    2022-03-17    45
    Length: 45, dtype: int64
    

    Example: larger curve size
    --------------------------
    >>> dd = df_roll_off(chain, loader = loader, n = 4)
    >>> dd.data[dt(2000):dt(0)]

                   0     1     2     3
    2000-01-01   0.0   1.0   2.0   3.0
    2000-01-02   0.0   1.0   2.0   3.0
             ...   ...   ...   ...
    2022-09-05  45.0  46.0  47.0  48.0
    2022-09-06  45.0  46.0  47.0  48.0

    """
    chain = dictable(chain)
    if roll not in chain.keys():
        chain[roll] = None
    if _data not in chain:
        chain[_data] = None
    data_ok = data is not None and len(data) > 0
    if data_ok and n == 1:
        if len(data.shape) == 2 and data.shape[1] > 1:
            data = data.iloc[:,0]
    if data_ok and n > 1:
        if len(data.shape) == 1 or data.shape[1] < n:
            data_ok = False
        elif data.shape[1] > n:
            data = data.iloc[:,:n]
    if data_ok:
        data_cutoff = data.index[-1]
        if cutoff is not None:
            cutoff = dt(cutoff)
            if cutoff < data_cutoff:
                data_cutoff = cutoff
                data = df_slice(data, ub = data_cutoff, openclose = '[]')
    else:
        data_cutoff = None

    if data_ok:
        old_data = data
        old_chain = chain[[row[roll] is not None and row[roll]<data_cutoff for row in chain]]
        chain = chain[[row[roll] is None or row[roll]>=data_cutoff for row in chain]]
    else:
        old_data = None
        old_chain = None

    load_on = as_list(load_on)
    size = len(chain)

    ## we load the data, loading only live data
    now = dt(0)
    loaded_data = []
    i = j = 0
    while j < size and i < n:
        row = chain[j]
        roll_date = row[roll]
        if roll_date is not None and data_cutoff is not None and roll_date < data_cutoff : ## old data
            loaded = row[_data]
        else: ## we load the data
            if len(load_on) == 0:
                loaded = row[loader]
            else:
                loaded = loader(**row[load_on])
            if (roll_date is None or roll_date >= now) and loaded is not None and len(loaded) > 0 and (data_cutoff is None or loaded.index[-1] > data_cutoff):
                i = i + 1
        loaded_data.append(loaded)
        j = j + 1
    if j < size:
        loaded_data.extend([None] * (size - j))
    chain[_data] = loaded_data

    ## we now determine missing rolling dates
    expiry = dt(expiry)
    rolls = chain[roll]
    lb = None
    for j in range(size-1, -1, -1):
        loaded = loaded_data[j]
        if rolls[j] is None and loaded is None:
            rolls[j] = lb
        elif loaded is not None and len(loaded) > 0:
            lb = loaded.index[0]
            if loaded.index[-1] < expiry and rolls[j] is None:
                rolls[j] = loaded.index[-1]
    chain[roll] = rolls

    ### we now begin the roll in earnest
    c = chain.inc(lambda data: data is not None and len(data) > 0)
    if len(c) == 0: # no new data
        return dictattr(chain = old_chain + chain, data = old_data, roll = roll)    
    ub = [_min(row[roll], row[_data].index[-1]) for row in c]
    if transform is not None:
        c = c.do(transform, _data)
    new_data = df_slice(c[_data], ub = ub, openclose = '(]', n = n)
    if old_data is not None and len(old_data) > 0:
        new_data = df_slice(new_data, lb = old_data.index[-1], openclose = '(]')
        res = pd.concat([old_data, new_data])
    else:
        res = new_data
    if old_chain is not None:
        chain = old_chain + chain
    chain = chain - _data
    return dictattr(data = res, chain = chain, roll = roll)

df_roll_off.output = ['data', 'roll', 'chain']