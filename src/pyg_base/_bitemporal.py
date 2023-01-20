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
_updated = 'updated'

def is_bi(df):
    return is_df(df) and _updated in df.columns

_nth = lambda v, n: v.iloc[min(n, len(v)-1)] if n>=0 else v.iloc[max(n, -len(v))]
    
def _as_what(what):
    if is_int(what):
        return partial(_nth, n = what)
    else:
        return what
    
    
def bi_read(df, asof = None, what = -1):
    """
    Parameters
    ----------
    df : a timeseries
        an item we can read bitemporally    
        
    asof: datetime
        read the values as if date is asof
        
    what: int, str or function
        how to select a value from multiple values associated with same index
        'first'  or 'last' would be the first non-nan values.
        0 : first actual value that was observed, even if nan
        -1: last actual value observed, even if nan
        -2,-3...    : one or two before last if available, else first
        1,2...    : 2nd or 3rd releases if available, else last
    if what is 'all' then the raw data is returned
    
    Example:
    --------
        df = pd.DataFrame(dict(a = [1,2,3,4,5,6], 
            _updated = drange(-2) + drange(-1) + drange(0)), 
            index = [dt(-2)] * 3 + [dt(-1)] * 2 + [dt(0)])
    
    """
    if not is_bi(df) or what == 'all':
        return df
    if is_date(asof):
        df = df[df[_updated]<=asof]
    if is_bi(asof):
        df = df[df[_updated] <= asof.reindex(df.index)[_updated]]
    index_name = df.index.name
    if len(df):        
        if index_name is None:
            df.index.name = 'index'
        gb = df.sort_values(_updated).groupby(df.index.name)
        res = gb.apply(_as_what(what)) ## since first and last return NON NAN VALUES, we need to override them
    else:
        res = df
    res = res.drop(columns = _updated)
    res.index.name = index_name
    if _columns in res.columns: ## we have a data frame with potentially mixed columns
        combos = list(set(res[_columns].values))
        if len(combos) == 1:
            columns = list(combos[0])
        else:
            columns = list(set(sum(combos, ())))
        res = res[columns]
    else:
        columns = res.columns
    if res.shape[1] == 1 and res.columns[0] == _series:
        res = res[_series]
    else:
        columns = [0 if c == _series else c for c in columns] ## this will clash with a dataframe with 0 column but thats fine
        res.columns = columns
    return res

def bi_asof(df, what = -1):
    """
    returns a timeseries stamped on the dates for which we have observations

    Parameters:
    ----------
    df : pandas dataframe
        bitemporal data
        
    what: str/int 
        how to aggregate multiple observations on the same dates
    

    Example:
    ----------
    >>> df = pd.DataFrame(dict(a = [1,2,3,4,5,6], 
                               updated = drange(-2) + drange(-1) + drange(0)), 
                               index = [dt(-2)] * 3 + [dt(-1)] * 2 + [dt(0)])
    >>> df

                a    updated
    2023-01-18  1 2023-01-18
    2023-01-18  2 2023-01-19
    2023-01-18  3 2023-01-20
    2023-01-19  4 2023-01-19
    2023-01-19  5 2023-01-20
    2023-01-20  6 2023-01-20
    
    >>> bi_asof(df, -1)
    
                a
    updated      
    2023-01-18  1
    2023-01-19  4
    2023-01-20  6
    
    >>> bi_read(df, what = -1)
    

    """
    if not is_bi(df) or what == 'all':
        return df
    gb = df.sort_index().groupby(_updated)
    res = gb.apply(_as_what(what)) ## since first and last return NON NAN VALUES, we need to override them
    res = res.drop(columns = _updated)
    return res


def _drop_repeats(d):
    """ 
    This is applied to a dataframe where 
    - all the observed index values are the same.
    - the _updated column is already sorted
    
    If there are partial bad values, we ffill, to ensure, the new observation are not nan
    Conversely, if the old value is nan, we accept the new value
    
    1) we first drop repeated values
    2) if we have a NEW value but the SAME _updated, we keep the later value. 
    
    """
    no_updated = d.drop(columns = _updated).ffill()
    old_values = no_updated.iloc[:-1].values
    new_values = no_updated.iloc[1:].values
    repeats = new_values == old_values
    res = d[~np.concatenate([[False],repeats.min(axis=1)])]
    res = res.drop_duplicates(subset = [_updated], keep = 'last')
    return res
 

@loop(list)
def _get_columns(df):
    return tuple(sort([col for col in df.columns if col!=_updated]))


@loop(list)
def _add_columns(df):
    if _columns in df:
        return df
    else:
        df = df.copy()
        columns = tuple([col for col in df.columns if col!=_updated])
        df[_columns] = [columns]*len(df)
    return df
    
@loop(list)
def _set_unique_column(df, col = _series):
    df = df.copy()
    columns = [col if c!=_updated else c for c in df.columns]
    df.columns = columns    
    return df

def bi_merge(old_data, new_data, asof = 'now', existing_data = None):
    """
    merges two or more bitemporal tables.
    
    Parameters:
    -----------
    old_data: old values, can be None, dataframe or bitemporal
    
    new_data: dataframe or list of dataframes
    
    asof:
        policy for handling non-bitemporal new data
        If new_data is NOT bitemporal converts using asof
    
    existing_data:
        policy for handling old_data
        'ignore/overwrite': ignores old data
        False/0/None: ignore non-bitemporals
        other values: use these to convert    
    
    Example: new values override old ones on same index and asof
    --------
    >>> from pyg import * 
    >>> s = pd.Series([1,2,3], drange(-3,-1))
    >>> old_data = Bi(s, 0)    
    >>> new_data = Bi(s+1, 0)
    >>> asof = 'now'; existing_data = None
    >>> merged = bi_merge(old_data, new_data)
    >>> assert eq(merged, new_data)
    >>> assert eq(bi_read(merged), s+1)
    
    Example: handling of nan
    ------------------------
    >>> s = pd.Series([1,2,3], drange(-3,-1))
    >>> s1 = pd.Series([1,2,np.nan], drange(-3,-1))
    >>> s2 = pd.Series([1,np.nan,3], drange(-3,-1))
    >>> b1 = Bi(s1, 0)     
    >>> b2 = Bi(s2, 1)
    >>> merged = bi_merge(b1, b2)
    >>> assert eq(bi_read(merged), s)
    >>> assert eq(bi_read(merged, dt(-1)), s1)
    >>> assert eq(bi_read(merged, what = 0), s1)
    >>> assert eq(bi_read(merged, dt(0)), s)
    
    Example: handling of multiple column names with pseudo-series
    ------------------------------------------
    ## if we have SINGLE COLUMNS dataframes, even if they clash by name, we merge into a SERIES    
    >>> from pyg import * 
    >>> s = pd.Series([1,2,3], drange(-3,-1))
    >>> p1 = pd.DataFrame(dict(a = [1,2,4]), drange(-3,-1))
    >>> p2 = pd.DataFrame(dict(b = [1,3,4]), drange(-3,-1))
    >>> bs = Bi(s, 0)
    >>> bp1 = Bi(p1, 1)
    >>> bp2 = Bi(p2, 2)
    >>> m = bi_merge(bs, bp1)
    >>> assert sort(m.columns) == sort([_series, _updated])
    >>> m = bi_merge(bs, [bp1, bp2])    
    >>> assert sort(m.columns) == sort([_series, _updated])
    >>> assert eq(bi_read(m), as_series(p2))
    
    Example: handling of multiple column names with a 2d dataframe
    ------------------------------------------
    >>> from pyg import * 
    >>> from pyg_base._bitemporal import _updated, _series, _columns
    >>> s = pd.Series([1,2,3], drange(-2))
    >>> df = pd.DataFrame(dict(a = [1,2,4], b = [4,5,6]), drange(-2))
    >>> bs = Bi(s, dt(1))
    >>> bdf = Bi(df, dt(2))
    >>> m = bi_merge(bs, bdf)
    >>> assert sort(m.columns) == sort(['a','b',_updated, _series, _columns])
    >>> assert eq(bi_read(m, asof = dt(1)), s)
    >>> assert eq(bi_read(m, asof = dt(2)), df)
    >>> df2 = pd.DataFrame(dict(a = [1,2,4], b = [4,5,6]), drange(-1,1))
    >>> bdf2 = Bi(df2, dt(2))
    >>> m2 = bi_merge(bs, bdf2)
    >>> assert sort(bi_read(m2).columns) == sort(('b', 0, 'a'))

    """
    if existing_data in ('ignore', 'overwrite'):
        old_bis = []
    else:
        if not existing_data:
            old_bis = [b for b in as_list(old_data) if is_bi(b)]
        else:
            old_bis = [b if is_bi(b) else Bi(b, existing_data) for b in as_list(old_data)]
    new_bis = [b if is_bi(b) else Bi(b, asof) for b in as_list(new_data)]            
    bis = old_bis + new_bis
    if len(bis) == 0:
        return None
    if len(bis) == 1:
        return bis[0]
    bis = [b if is_bi(b) else Bi(b, asof) for b in bis]
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
    gb = df.sort_values(_updated).groupby(df.index.name)
    res = pd.concat([_drop_repeats(d) for _, d in gb])
    res.index.name = index_name
    return res
    

def Bi(df, asof = None):
    """
    Creates a bitemporal dataframe
    
    :Example: make the _updated same as date index
    ---------
    >>> df = pd.DataFrame(dict(a = range(11)), drange(-10))
    >>> assert eq(Bi(df, 0)[_updated].values, df.index.values) 

    :Example: make the _updated right now...
    ---------
    >>> assert len(set(Bi(df)[_updated].values)) == 1

    :Example: make the _updated the 3am of next business...
    --------- 
    >>> Bi(df, ['1b', '3h'])
                 a               _updated
    2022-10-15   0 2022-10-18 03:00:00
    2022-10-16   1 2022-10-18 03:00:00
    ....
    2022-10-24   9 2022-10-25 03:00:00
    2022-10-25  10 2022-10-26 03:00:00

    :Example: make the _updated any exact date...
    ---------
    >>> asof = dt(2)
    >>> assert set(Bi(df, asof)[_updated]) ==  set([asof])
    
    """
    if asof is None or is_bi(df):
        return df
    if is_series(df):
        df = pd.DataFrame(df, columns = [_series])
    else:
        df = df.copy()
    if asof == 'shift':
        now = dt()
        df[_updated] = list(df.index[1:]) + [now]        
    elif is_bump(asof):
        now = dt()
        df[_updated] = dt_bump(df, asof).index
        df.loc[(df[_updated] > now), _updated] = now
    elif isinstance(asof, list):
        now = dt()
        df[_updated] = dt_bump(df, *asof).index
        df.loc[(df[_updated] > now), _updated] = now
    else:
        df[_updated] = dt(asof)
    return df

