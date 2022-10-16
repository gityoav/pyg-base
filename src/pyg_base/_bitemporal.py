### we create an interface for read/write of pandas which is bitemporal
# the data
from pyg_base import is_df, dt, is_date, is_series, is_bump, dt_bump, is_int, as_list
import pandas as pd
import numpy as np
from functools import partial 

_series = '_is_series'
_asof = '_asof'

def is_bi(df):
    return is_df(df) and _asof in df.columns

_first = lambda v: v.iloc[0]
_last = lambda v: v.iloc[-1]
_nth = lambda v, n: v.iloc[min(n, len(v)-1)] if n>=0 else v.iloc[max(n, -len(v))]
    
def _as_what(what):
    if what == 'last':
        return _last
    elif what == 'first':
        return _first
    elif is_int(what):
        return partial(_nth, n = what)
    else:
        return what
    
    

def bi_read(df, asof = None, what = 'last'):
    """
    Parameters
    ----------
    df : a timeseries
        an item we can read bitemporally    
        
    asof: datetime
        read the values as if date is asof
        
    what: str/function
        how to handle multiple values...
    
    Example:
    --------
        df = pd.DataFrame(dict(a = [1,2,3,4,5,6], 
            _asof = drange(-2) + drange(-1) + drange(0)), 
            index = [dt(-2)] * 3 + [dt(-1)] * 2 + [dt(0)])
    
    """
    if not is_bi(df):
        return df
    if is_date(asof):
        df = df[df[_asof]<=asof]
    if is_bi(asof):
        df = df[df[_asof] <= asof.reindex(df.index)[_asof]]
    index_name = df.index.name
    if index_name is None:
        df.index.name = 'index'
    gb = df.sort_values(_asof).groupby(df.index.name)
    res = gb.apply(_as_what(what)) ## since first and last return NON NAN VALUES, we need to override them
    res = res.drop(columns = _asof)
    res.index.name = index_name
    if res.shape[1] == 1 and res.columns[0] == _series:
        res = res[_series]
    return res




def bi_updates(new, old):
    """
    >>> from pyg import * 
    >>> old = pd.DataFrame(dict(a = [1,2,3,4,5,6], 
            _asof = drange(-2) + drange(-1) + drange(0)), 
            index = [dt(-2)] * 3 + [dt(-1)] * 2 + [dt(0)])

    >>> new = pd.DataFrame(dict(a = [7,8,3,10],
                                _asof = drange(-2,1)), 
                           index = drange(-4,-1))
    """

    raw = new.drop(columns = _asof)
    prev = bi_read(old, asof = new, what = 'last')
    update = new[~(prev.reindex(raw.index) == raw).min(axis=1)]
    return update

def _drop_repeats(d):
    """ 
    This is applied to a dataframe where 
    - all the observed index values are the same.
    - the _asof column is already sorted
    
    If there are partial bad values, we ffill, to ensure, the new observation are not nan
    Conversely, if the old value is nan, we accept the new value
    
    1) we first drop repeated values
    2) if we have a NEW value but the SAME _asof, we keep the later value. 
    
    """
    no_asof = d.drop(columns = _asof).ffill()
    old_values = no_asof.iloc[:-1].values
    new_values = no_asof.iloc[1:].values
    repeats = new_values == old_values
    res = d[~np.concatenate([[False],repeats.min(axis=1)])]
    res = res.drop_duplicates(subset = [_asof], keep = 'last')
    return res
 

def bi_merge(*bis):
    """
    Example: new values override old ones on same index and asof
    --------
    >>> from pyg import * 
    >>> s = pd.Series([1,2,3], drange(2))
    >>> old = Bi(s, 0)    
    >>> new = Bi(s+1, 0)
    >>> merged = bi_merge(old, new)
    >>> assert eq(merged, new)
    >>> assert eq(bi_read(merged), s+1)
    
    Example: handling of nan
    ------------------------
    >>> s = pd.Series([1,2,3], drange(2))
    >>> s1 = pd.Series([1,2,np.nan], drange(2))
    >>> s2 = pd.Series([1,np.nan,3], drange(2))
    >>> b1 = Bi(s1, 0)     
    >>> b2 = Bi(s2, 1)
    >>> merged = bi_merge(b1, b2)
    >>> assert eq(bi_read(merged), s)
    >>> assert eq(bi_read(merged, dt(2)), s1)
    >>> assert eq(bi_read(merged, what = 'first'), s1)
    >>> assert eq(bi_read(merged, dt(3)), s)

    """
    bis = as_list(bis)
    df = pd.concat(bis)
    index_name = df.index.name
    if index_name is None:
        df.index.name = 'index'
    gb = df.sort_values(_asof).groupby(df.index.name)
    res = pd.concat([_drop_repeats(d) for _, d in gb])
    res.index.name = index_name
    return res
    

def Bi(df, asof = None):
    """
    Creates a bitemporal dataframe
    
    :Example: make the _asof same as date index
    ---------
    >>> df = pd.DataFrame(dict(a = range(11)), drange(10))
    >>> assert eq(Bi(df, 0)[_asof].values, df.index.values) 

    :Example: make the _asof right now...
    ---------
    >>> assert len(set(Bi(df)[_asof].values)) == 1

    :Example: make the _asof the 3am of next business...
    --------- 
    >>> Bi(df, ['1b', '3h'])
                 a               _asof
    2022-10-15   0 2022-10-18 03:00:00
    2022-10-16   1 2022-10-18 03:00:00
    ....
    2022-10-24   9 2022-10-25 03:00:00
    2022-10-25  10 2022-10-26 03:00:00

    :Example: make the _asof any exact date...
    ---------
    >>> asof = dt(2)
    >>> assert set(Bi(df, asof)[_asof]) ==  set([asof])
    
    """
    if is_series(df):
        df = pd.DataFrame(df, columns = [_series])
    else:
        df = df.copy()
    if is_bump(asof):
        df[_asof] = dt_bump(df, asof).index
    elif isinstance(asof, list):
        df[_asof] = dt_bump(df, *asof).index
    else:
        df[_asof] = dt(asof)
    return df

