### we create an interface for read/write of pandas which is bitemporal
# the data
from pyg_base._types import is_df, is_date, is_series, is_int
from pyg_base._dates import dt, dt_bump, is_bump
from pyg_base._as_list import as_list
from pyg_base._loop import loop
from pyg_base._sort import sort

import pandas as pd
import numpy as np
from functools import partial 

_series = '_is_series'
_columns = '_column_names'
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
        
    what: int, str or function
        how to select a value from multiple values associated with same index
        0 or 'first' : first value that was observed
        -1 or 'last' : last value observed
        -2,-3...    : one or two before last if available, else first
        1,2...    : 2nd or 3rd releases if available, else last
    
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
    if _columns in res.columns: ## we have a data frame with potentially mixed columns
        combos = list(set(res[_columns].values))
        if len(combos) == 1:
            columns = list(combos[0])
        else:
            columns = list(set(sum(combos, ())))
        res = res[columns]
        columns = [0 if c == _series else c for c in columns]
        res.columns = columns
    res.index.name = index_name
    if res.shape[1] == 1 and res.columns[0] == _series:
        res = res[_series]
    return res


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
 

@loop(list)
def _get_columns(df):
    return tuple(sort([col for col in df.columns if col!=_asof]))


@loop(list)
def _add_columns(df):
    if _columns in df:
        return df
    else:
        df = df.copy()
        columns = tuple([col for col in df.columns if col!=_asof])
        df[_columns] = [columns]*len(df)
    return df
    
@loop(list)
def _set_unique_column(df, col = _series):
    df = df.copy()
    columns = [col if c!=_asof else c for c in df.columns]
    df.columns = columns    
    return df

def bi_merge(*bis):
    """
    merges two or more bitemporal tables
    
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
    
    Example: handling of multiple column names with pseudo-series
    ------------------------------------------
    ## if we have SINGLE COLUMNS dataframes, even if they clash by name, we merge into a SERIES    
    >>> from pyg import * 
    >>> s = pd.Series([1,2,3], drange(2))
    >>> p1 = pd.DataFrame(dict(a = [1,2,4]), drange(2))
    >>> p2 = pd.DataFrame(dict(b = [1,3,4]), drange(2))
    >>> bs = Bi(s, 0)
    >>> bp1 = Bi(p1, 1)
    >>> bp2 = Bi(p2, 2)
    >>> m = bi_merge(bs, bp1)
    >>> assert sort(m.columns) == sort([_series, _asof])
    >>> m = bi_merge(bs, bp1, bp2)    
    >>> assert sort(m.columns) == sort([_series, _asof])
    >>> assert eq(bi_read(m), as_series(p2))
    
    Example: handling of multiple column names with a 2d dataframe
    ------------------------------------------
    >>> from pyg import * 
    >>> from pyg_base._bitemporal import _asof, _series, _columns
    >>> s = pd.Series([1,2,3], drange(-2))
    >>> df = pd.DataFrame(dict(a = [1,2,4], b = [4,5,6]), drange(-2))
    >>> bs = Bi(s, dt(1))
    >>> bdf = Bi(df, dt(2))
    >>> m = bi_merge(bs, bdf)
    >>> assert sort(m.columns) == sort(['a','b',_asof, _series, _columns])
    >>> assert eq(bi_read(m, asof = dt(1)), s)
    >>> assert eq(bi_read(m, asof = dt(2)), df)
    >>> df2 = pd.DataFrame(dict(a = [1,2,4], b = [4,5,6]), drange(-1,1))
    >>> bdf2 = Bi(df2, dt(2))
    >>> m2 = bi_merge(bs, bdf2)
    >>> assert sort(bi_read(m2).columns) == sort(('b', 0, 'a'))

    """
    bis = as_list(bis)
    ### we first determine columns needed
    with_columns = [b for b in bis if _columns in b.columns]
    if len(with_columns):
        bis = _add_columns(bis)
    else:
        columns = list(set(_get_columns(bis)))
        if len(columns) > 1:
            ns = [len(c) for c in columns]
            if max(ns) > 1: ## too bad
                bis = _add_columns(bis)
            else: ## we remove all column names
                bis = _set_unique_column(bis)
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

