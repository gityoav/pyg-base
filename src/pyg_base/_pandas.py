"""
We want to simplify the operations for pandas dataframes assuming we are using timeseries as the main objects.

When we have multiple timeseries, we will:
    
    1) calculate joint index using df_index()
    2) reindex each timeseries to the joint index
    
We then need to worry about multiple columns if there are. If none, each timeseries will be considered as pd.Series

If there are multiple columns, we will perform the calculations columns by columns. 

"""
from pyg_base._types import is_df, is_str, is_num, is_ts, is_series, is_tss, is_int, is_arr, is_ts, is_arrs, is_tuples, is_pd, is_date
from pyg_base._dictable import dictable
from pyg_base._as_list import as_list
from pyg_base._zip import zipper
from pyg_base._reducer import reducing, reducer
from pyg_base._decorators import  wrapper
from pyg_base._loop import loop
from pyg_base._dates import dt
from pyg_base._txt import lower
import pandas as pd
import numpy as np
from functools import partial
from copy import copy
import inspect
import datetime


__all__ = ['df_fillna', 'df_index', 'df_reindex', 'df_columns', 'presync', 'np_reindex', 'nona', 'df_slice', 'df_unslice', 'min_', 'max_', 'add_', 'mul_', 'sub_', 'div_', 'pow_']

def _list(values):
    """
    >>> assert _list([1,2,[3,4,5,[6,7]],dict(a =[8,9], b=[10,[11,12]])])  == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  
    >>> assert _list(1)  == [1]  
    >>> assert _list(dict(a=1, b=2))  == [1,2]  

    """
    if isinstance(values, list):
        return sum([_list(df) for df in values], [])
    elif isinstance(values, dict):
        return _list(list(values.values()))
    else:
        return [values]


@loop(list, tuple, dict)
def _index(ts):
    if isinstance(ts, pd.Index):
        return ts
    elif is_pd(ts):
        return ts.index
    elif is_arr(ts):
        return len(ts)
    else:
        raise ValueError('did not provide an index')
    

def _df_index(indexes, index):
    if len(indexes) > 0:
        if is_str(index):
            if index[0].lower() == 'i':#nner
                return reducing('intersection')(indexes)        
            elif index[0].lower() == 'o':#uter
                return reducing('union')(indexes)        
            elif index[0].lower() == 'l':#uter
                return indexes[0]
            elif index[0].lower() == 'r':#uter
                return indexes[-1]
        else:
            return _index(index)
    else:
        return None


def _np_index(indexes, index):
    if len(indexes) > 0:
        if index[0].lower() == 'i':#nner
            return min(indexes)        
        elif index[0].lower() == 'o':#uter
            return max(indexes)        
        elif index[0].lower() == 'l':#uter
            return indexes[0]
        elif index[0].lower() == 'r':#uter
            return indexes[-1]
    else:
        return None


def df_index(seq, index = 'inner'):
    """
    Determines a joint index of multiple timeseries objects.

    :Parameters:
    ----------------
    seq : sequence whose index needs to be determined
        a (possible nested) sequence of timeseries/non-timeseries object within lists/dicts
    index : str, optional
        method to determine the index. The default is 'inner'.

    :Returns:
    -------
    pd.Index
        The joint index.
        
    :Example:
    ---------
    >>> tss = [pd.Series(np.random.normal(0,1,10), drange(-i, 9-i)) for i in range(5)]
    >>> more_tss_as_dict = dict(zip('abcde',[pd.Series(np.random.normal(0,1,10), drange(-i, 9-i)) for i in range(5)]))
    >>> res = df_index(tss + [more_tss_as_dict], 'inner')
    >>> assert len(res) == 6
    >>> res = df_index(more_tss_as_dict, 'outer')
    >>> assert len(res) == 14
    """
    listed = _list(seq)
    indexes = [ts.index for ts in listed if is_pd(ts)]
    if len(indexes):
        return _df_index(indexes, index)
    arrs = [len(ts) for ts in listed if is_arr(ts)]
    if len(arrs):
        return _np_index(arrs, index)
    else:
        return None
    

def df_columns(seq, index = 'inner'):
    """
    returns the columns of the joint object
    
    :Example:
    ---------
    >>> a = pd.DataFrame(np.random.normal(0,1,(100,5)), drange(-99), list('abcde'))
    >>> b = pd.DataFrame(np.random.normal(0,1,(100,5)), drange(-99), list('bcdef'))
    >>> assert list(df_columns([a,b])) == list('bcde')
    >>> assert list(df_columns([a,b], 'oj')) == list('abcdef')
    >>> assert list(df_columns([a,b], 'lj')) == list('abcde')
    >>> assert list(df_columns([a,b], 'rj')) == list('bcdef')

    :Parameters:
    ----------
    seq : sequence of dataframes 
        DESCRIPTION.
    index : str, optional
        how to inner-join. The default is 'inner'.

    :Returns:
    -------
    pd.Index
        list of columns.
    """
    
    listed = _list(seq)
    indexes= [ts.columns for ts in listed if is_df(ts) and ts.shape[1]>1 and len(set(ts.columns)) == ts.shape[1]] #dataframe with non-unique columns are treated like arrays
    if len(indexes):
        return _df_index(indexes, index)
    arrs = [ts.shape[1] for ts in listed if (is_arr(ts) or is_df(ts)) and len(ts.shape)>1 and ts.shape[1]>1]
    if len(arrs):
        return _np_index(arrs, index)
    return None

@loop(list, tuple, dict)
def _df_fillna(df, method = None, axis = 0, limit = None):
    methods = as_list(method)
    if len(methods) == 0:
        return df
    if is_arr(df):
        return df_fillna(pd.DataFrame(df) if len(df.shape)==2 else pd.Series(df), method, axis, limit).values
    res = df
    for m in methods:
        if is_num(m):
            res = res.fillna(value = m, axis = axis, limit = limit)
        elif m in ['backfill', 'bfill', 'pad', 'ffill']:
            res = res.fillna(method = m, axis = axis, limit = limit)
        elif m in ['ffill_na', 'ffill_0']: # forward fill but only up to the end of a timeseries    
            invalid = np.nan if m == 'ffill_na' else 0.
            if len(df.shape) == 1:
                last_valid = df.last_valid_index()
                if last_valid is not None:    
                    res = res.fillna(method = 'ffill', axis = axis, limit = limit)
                    res[res.index>last_valid] = invalid
            else:
                res = pd.concat([_df_fillna(res.iloc[:, i], method, axis, limit) for i in range(res.shape[1])], axis=1)
        elif is_date(m):
            res = res.fillna(method = 'ffill', axis = axis, limit = limit)
            res[res.index>m] = np.nan
        elif m in ('fnna', 'nona'):
            nonan = ~np.isnan(res)
            if len(res.shape)==2:
                nonan = nonan.max(axis=1)
            if m == 'fnna':
                nonan = nonan[nonan.values]
                if len(nonan):
                    res = res[nonan.index[0]:]
                else:
                    res = res.iloc[:0]
            elif m == 'nona':
                res = res[nonan.values]
        else:
            if is_num(limit) and limit<0:
                res = res.interpolate(method = m, axis = axis, limit = abs(limit), 
                                      limit_direction = 'backward')
            else:
                res = res.interpolate(method = m, axis = axis, limit = limit)
    return res

def df_fillna(df, method = None, axis = 0, limit = None):
    """
    Equivelent to df.fillna() except:

    - support np.ndarray as well as dataframes
    - support multiple methods of filling/interpolation
    - supports removal of nan from the start/all of the timeseries
    - supports action on multiple timeseries
    
    :Parameters:
    ----------------
    df : dataframe/numpy array
        
    method : string, list of strings or None, optional
        Either a fill method (bfill, ffill, pad)
        Or an interplation method: 'linear', 'time', 'index', 'values', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'barycentric', 'krogh', 'spline', 'polynomial', 'from_derivatives', 'piecewise_polynomial', 'pchip', 'akima', 'cubicspline'
        Or 'fnna': removes all to the first non nan
        Or 'nona': removes all nans
        Or 'ffill_na': forward fills all the way to last valid index and then stops
        Or 'ffill_0': forward fills all the way to last valid index and then fills with zero [Think about a position in an option that expires]
    axis : int, optional
        axis. The default is 0.
    limit : TYPE, optional
        when filling, how many nan get filled. The default is None (indefinite)
    
    :Example: method ffill or bfill
    -----------------------------------------------
    >>> from pyg import *; import numpy as np
    >>> df = np.array([np.nan, 1., np.nan, 9, np.nan, 25])    
    >>> assert eq(df_fillna(df, 'ffill'), np.array([ np.nan, 1.,  1.,  9.,  9., 25.]))
    >>> assert eq(df_fillna(df, ['ffill','bfill']), np.array([ 1., 1.,  1.,  9.,  9., 25.]))
    >>> assert eq(df_fillna(df, ['ffill','bfill']), np.array([ 1., 1.,  1.,  9.,  9., 25.]))

    >>> df = np.array([np.nan, 1., np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9, np.nan, 25])    
    >>> assert eq(df_fillna(df, 'ffill', limit = 2), np.array([np.nan,  1.,  1.,  1., np.nan, np.nan, np.nan, np.nan,  9.,  9., 25.]))

    df_fillna does not maintain state of latest 'prev' value: use ffill_ for that.

    :Example: interpolation methods
    -----------------------------------------------
    >>> from pyg import *; import numpy as np
    >>> df = np.array([np.nan, 1., np.nan, 9, np.nan, 25])    
    >>> assert eq(df_fillna(df, 'linear'), np.array([ np.nan, 1.,  5.,  9.,  17., 25.]))
    >>> assert eq(df_fillna(df, 'quadratic'), np.array([ np.nan, 1.,  4.,  9.,  16., 25.]))


    :Example: method = fnna and nona
    ---------------------------------------------
    >>> from pyg import *; import numpy as np
    >>> ts = np.array([np.nan] * 10 + [1.] * 10 + [np.nan])
    >>> assert eq(df_fillna(ts, 'fnna'), np.array([1.]*10 + [np.nan]))
    >>> assert eq(df_fillna(ts, 'nona'), np.array([1.]*10))

    >>> assert len(df_fillna(np.array([np.nan]), 'nona')) == 0
    >>> assert len(df_fillna(np.array([np.nan]), 'fnna')) == 0

    :Returns:
    -------
    array/dataframe with nans removed/filled

    """
    return _df_fillna(df, method = method, axis = axis, limit = limit)

@loop(dict, list, tuple)
def _nona(df, value = np.nan, edge = None):
    if np.isnan(value):
        mask = np.isnan(df)
    elif np.isinf(value):
        mask = np.isinf(df)
    else:
        mask = df == value
    while len(mask.shape) > 1:
        mask = mask.min(axis = 1)
    res = df[~mask]
    if edge is None or len(res) == 0 or not is_pd(df):
        return res
    elif edge == 1: ## cut only latest values
        return df_slice(df, ub = res.index[-1], openclose = '[]')
    elif edge == -1: ## cut only historic values
        return df_slice(df, lb = res.index[0], openclose = '[]')
    

def nona(a, value = np.nan, edge = None):
    """
    removes rows that are entirely nan (or a specific other value)

    :Parameters:
    ----------------
    a : dataframe/ndarray
        
    value : float, optional
        value to be removed. The default is np.nan.
        
    edge: None/1/-1
        remove all nans/remove latest nan only/remove historic nan only
        
    :Example:
    ----------
    >>> from pyg import *
    >>> a = np.array([1,np.nan,2,3])
    >>> assert eq(nona(a), np.array([1,2,3]))
    >>> assert eq(nona(a, edge = 1), a)

    >>> df = pd.Series([np.nan,1,np.nan,2,3,np.nan])
    >>> assert eq(nona(df, edge = 1), pd.Series([np.nan,1,np.nan,2,3]))
    >>> assert eq(nona(df, edge = -1), pd.Series([1,np.nan,2,3,np.nan], range(1,6)))
    
    :Example: multiple columns
    ---------------------------
    >>> a = np.array([[1,np.nan,2,np.nan], [np.nan, np.nan, np.nan, 3]]).T 
    >>> b = np.array([[1,2,np.nan], [np.nan, np.nan, 3]]).T ## 2nd row has nans across
    >>> assert eq(nona(a), b)


    """
    return _nona(a, value = value, edge = edge)


@loop(list, tuple, dict)
def _df_reindex(ts, index, method = None, limit = None):
    methods = as_list(method)
    if is_pd(ts):
        if is_int(index):
            raise ValueError('trying to reindex dataframe %s using numpy interval length %i'%(ts, index))
        if len(methods) and methods[0] in ['backfill', 'bfill', 'pad', 'ffill']:
            res = _nona(ts).reindex(index, method = methods[0], limit = limit)
            res = _df_fillna(res, method = methods[1:], limit = limit)
        else:
            res = ts.reindex(index)
            res = _df_fillna(res, method = method, limit = limit)
        return res
    elif is_arr(ts):
        if isinstance(index, pd.Index):
            if len(index) == len(ts) or len(ts) <= 1:
                return ts
            else:
                raise ValueError('trying to reindex numpy array %s using pandas index %s'%(ts, index))
        elif is_int(index):
            if index<len(ts):
                res = ts[-index:]
            elif index>len(ts):
                shape = (index - len(ts),) + ts.shape[1:]
                res = np.concatenate([np.full(shape, np.nan),ts])
            else:
                res = ts
            return df_fillna(res, method = methods, limit = limit)
        else:
            return ts
    else:
        return ts
    

@loop(list, tuple, dict)
def _df_recolumn(ts, columns):
    if columns is not None and is_df(ts) and ts.shape[1] > 1 and len(set(ts.columns)) == ts.shape[1]:
        return pd.DataFrame({col: ts[col].values if col in ts.columns else np.nan for col in columns}, index = ts.index)
    else:
        return ts

def df_recolumn(ts, columns = None):
    return _df_recolumn(ts, columns)

def np_reindex(ts, index, columns = None):
    """
    pyg assumes that when working with numpy arrays representing timeseries, you:
        - determine a global timestamp
        - resample all timeseries to that one, and then covert to numpy.array, possibly truncating leading nan's.
        - do the maths you need to do
        - having worked with numpy arrays, if we want to reindex them back into dataframe, use np_reindex
    
    :Example:
    -------
    >>> from pyg import *
    >>> ts = np.array(np.random.normal(0,1,1000))
    >>> index = pd.Index(drange(-1999))
    >>> np_reindex(ts, index)

    :Parameters:
    ----------------
    ts : numpy array

    index : pandas.Index

    columns: list/array of columns names

    :Returns:
    ----------
    pd.DataFrame/pd.Series

    """
    if is_pd(index):
        index = index.index
    if len(index)>len(ts):
        index = index[-len(ts):]
    elif len(index)<len(ts):
        ts = ts[-len(index):]
    res = pd.Series(ts, index) if len(ts.shape)<2 else pd.DataFrame(ts, index)
    if columns is not None:
        if is_df(columns):
            columns = columns.columns
        res.columns = columns
    return res

def df_reindex(ts, index = None, method = None, limit = None):
    """
    A slightly more general version of df.reindex(index)

    :Parameters:
    ----------------
    ts : dataframe or numpy array (or list/dict of theses)
        timeseries to be reindexed
    index : str, timeseries, pd.Index.
        The new index
    method : str, list of str, float, optional
        various methods of handling nans are available. The default is None.
        See df_fillna for a full list.

    :Returns:
    -------
    timeseries/np.ndarray (or list/dict of theses)
        timeseries reindex.
        
    :Example: index = inner/outer
    -----------------------------
    >>> tss = [pd.Series(np.random.normal(0,1,10), drange(-i, 9-i)) for i in range(5)]
    >>> res = df_reindex(tss, 'inner')
    >>> assert len(res[0]) == 6
    >>> res = df_reindex(tss, 'outer')
    >>> assert len(res[0]) == 14

    :Example: index provided
    -----------------------------
    >>> tss = [pd.Series(np.random.normal(0,1,10), drange(-i, 9-i)) for i in range(5)]
    >>> res = df_reindex(tss, tss[0])
    >>> assert eq(res[0], tss[0])
    >>> res = df_reindex(tss, tss[0].index)
    >>> assert eq(res[0], tss[0])

    """
    if index is None:
        return ts
    elif is_str(index):
        index = df_index(ts, index)
    elif is_ts(index):
        index = index.index
    elif is_arr(index):
        index = pd.Index(index)
    return _df_reindex(ts, index = index, method = method, limit = limit)


def _dtype(o):
    return int if is_int(o) else float if is_num(o) else 'object' if o is None or is_str(o) else None


def as_series(df, col = None, unique_column = False):
    """
    In pandas we have 
        - series
        - 2d dataframes 
        - pseudo-series: dataframes with one column and a name
    
    as_series converts pseudo-series/series into:
        - series if col is None
        - pseudo-series with header of col, if col is provided

    If multiple dfs are provided, here is how we handle the mixture:
        - if any proper 2d dfs are present, nothing is done
        
        - if we have a mixture of pseudo and series: all are converted to series
        - if we have multiple pseudo series with different headers: all converted to series
        - if we have multiple pseudo series with same headers: we keep the header if unique_column else series
        

    :Example:
    ---------
    >>> s = pd.Series([1,2,3], drange(2))
    >>> s2 = pd.Series([4,5,6], drange(2))
    >>> p = pd.DataFrame(dict(p=[1,2,3]), drange(2))
    >>> d = pd.DataFrame(dict(a=[1,2,3], b=[1,2,3]), drange(2))
    >>> p2 = pd.DataFrame(dict(p=[4,5,6]), drange(2))
    >>> pq = pd.DataFrame(dict(q=[1,2,3]), drange(2))

    >>> assert eq(as_series(p), s)
    >>> assert eq(as_series(s, 'p'), p)
    >>> assert eq(as_series([p,s]), [s,s]) ## mixture all convert to series
    >>> assert eq(as_series([p,p2]), [s,s2])
    >>> assert eq(as_series([p,p2], unique_column = True), [p,p2])
    >>> assert eq(as_series([s,p], 'q'), [pq,pq]) ## both converted to header q

    """
    if isinstance(df, list):
        proper_dfs = [d for d in df if is_df(d) and d.shape[1] > 1]
        if len(proper_dfs):
            return df
        else:
            if col is None and unique_column:
                cols = list(set([0 if is_series(d) else d.columns[0] for d in df if is_pd(d)]))
                if len(cols) == 1 and cols[0]!=0:
                    col = cols[0]                
            return type(df)([as_series(d, col) for d in df])
    if col is not None:
        columns = as_list(col)
        if is_series(df):
            return pd.DataFrame(df, columns = columns)
        elif is_df(df) and df.shape[1] == 1:
            df = df.copy()
            df.columns = columns
            return df
        else:
            return df
    else:        
        if is_df(df) and df.shape[1] == 1:
            return df[df.columns[0]]
        else:
            return df

def df_concat(objs, columns = None, axis = 1, join = 'outer', method = None, limit = None):
    """
    simple concatenator, 
    - defaults to to concatenating by date (for timeseries)
    - supports columns renaming

    :Parameters:
    ----------
    objs : list/dict
        collection of timeseries
    columns : str/list
        Names of new columns. The default is None.
    axis : int, optional
        axis to merge. The default is 1.
    join : str, optional
        join method inner/outer, see pd.concat. The default is 'outer'.

    :Returns:
    -------
    res : pd.DataFrame
        joined dataframe
        
    :Example:
    ---------
    >>> objs = [pd.Series([1,2,3], [4,5,6]), pd.Series([3,4,5], [1,2,4])]
    >>> columns = ['a', 'b']; 
    >>> axis = 1; join = 'outer'
    >>> res = df_concat(objs, columns)

    >>> res
    >>>      a    b
    >>> 1  NaN  3.0
    >>> 2  NaN  4.0
    >>> 4  1.0  5.0
    >>> 5  2.0  NaN
    >>> 6  3.0  NaN    

    >>> df_concat(res, dict(a = 'x', b = 'y'))
    >>> res
    >>>      x    y
    >>> 1  NaN  3.0
    >>> 2  NaN  4.0
    >>> 4  1.0  5.0
    >>> 5  2.0  NaN
    >>> 6  3.0  NaN    

    :Example: numpy arrays concatenation
    -----------------------
    >>> objs = [np.array([1,2,3]), np.array([4,5,6]), 3, 4]
    >>> df_concat(objs)

    Example: pd.Series and 1d pd.DataFrame handling for vertical axis=0 concatenation
    --------------------------------------------
    >>> from pyg import * 
    >>> s = pd.Series([1,2,3], drange(2))
    >>> p = pd.DataFrame(dict(p=[1,2,3]), drange(2))
    >>> pq = pd.DataFrame(dict(q=[1,2,3]), drange(2))

    >>> assert is_series(df_concat([s,p], axis = 0))
    >>> assert is_df(df_concat([s,p], columns = 'p', axis = 0))
    >>> assert is_series(df_concat([p,pq], axis = 0)) ## multiple columns, handled as series

    """
    if isinstance(objs, dict):
        columns = list(objs.keys())
        objs = list(objs.values())
    if isinstance(objs, (list, tuple)):
        df_objs = [o for o in objs if is_pd(o)]
        np_objs = [o for o in objs if is_arr(o)]
        if len(df_objs):
            if axis == 0:
                df_objs = as_series(df_objs, col = columns, unique_column=True)
            if len(df_objs) < len(objs): # we have a mixture of dfs and numpy
                if axis == 1:
                    res = pd.concat(df_objs, axis = axis, join = join)
                    df_objs = [o if is_pd(o) else pd.Series(o, res.index, dtype = _dtype(o)) for o in objs]
                    res = pd.concat(df_objs, axis = axis, join = join)
                else:
                    raise TypeError('cannot concatenate vertically dataframes and non-dataframes')
            else:
                res = pd.concat(df_objs, axis = axis, join = join)                
        elif len(np_objs):
            ns = set([o.shape[0] for o in np_objs])
            if len(ns) == 1:
                n = list(ns)[0]
            else:
                raise ValueError('numpy arrays must be of equal length, other joins not implemented')
            np_objs = [(o[:, np.newaxis] if len(o.shape)==1 else o) if is_arr(o) else np.full((n,1), o) for o in objs]
            res = np.concatenate(np_objs, axis = axis)
        else:
            return objs
    elif isinstance(objs, pd.DataFrame):
        res = objs.copy() if columns is not None else objs
    elif is_arr(objs):
        res = objs.copy()
    else:
        raise ValueError('not sure how to convert this into a single dataframe')
    if columns is not None and is_df(res):
        if isinstance(columns, list):
            res.columns = columns 
        elif is_str(columns):
            res.columns = as_list(columns)
        else:
            res = res.rename(columns = columns)
    res = df_fillna(res, method = method, axis = 1 if axis == 0 else 1, limit = limit)
    return res


@loop(list, dict, tuple)
def _df_column(ts, column, i = None, n = None):
    """
    This is mostly a helper function to help us loop through multiple columns.
    Function grabs a column from a dataframe/2d array

    :Parameters:
    ----------
    ts : datafrane
        the original dataframe or 2-d numpy array
    column : str
        name of the column to grab.
    i : int, optional
        Can grab the column using its index. The default is None.
    n : int, optional
        asserting the number of columns, ts.shape[1]. The default is None.

    :Returns:
    -------
    a series or a 1-d numpy array
    """
    
    if is_df(ts):
        if ts.shape[1] == 1:
            return ts[ts.columns[0]]
        elif column in ts.columns:
            return ts[column]
        elif column is None and i is not None:
            if len(set(ts.columns)) == ts.shape[1]: #unique columns, don't call me using i
                raise ValueError('trying to grab %ith column from a dataframe with proper columns: %s'%(i, ts.columns))
            elif n is not None and ts.shape[1]!=n:
                raise ValueError('trying to grab %ith column and asserting must have %i columns but have %i'%(i, n, ts.shape[1]))
            else:
                if i<ts.shape[1]:
                    return ts.iloc[:,i]
                else:
                    return np.nan
        else:
            return np.nan
    elif is_arr(ts) and len(ts.shape) == 2:
        if ts.shape[1] == 1:
            return ts.T[0]
        elif i is not None:
            if n is not None and ts.shape[1]!=n:
                raise ValueError('trying to grab %ith column and asserting must have %i columns but have %i'%(i, n, ts.shape[1]))
            elif i<ts.shape[1]:
                return ts.T[i]
            else:
                return np.nan
        else:
            return ts
    else:
        return ts


def df_column(ts, column, i = None, n = None):
    """
    This is mostly a helper function to help us loop through multiple columns.
    Function grabs a column from a dataframe/2d array

    :Parameters:
    ----------
    ts : datafrane
        the original dataframe or 2-d numpy array
    column : str
        name of the column to grab.
    i : int, optional
        Can grab the column using its index. The default is None.
    n : int, optional
        asserting the number of columns, ts.shape[1]. The default is None.

    :Returns:
    -------
    a series or a 1-d numpy array
    """
    return _df_column(ts = ts, column = column, i = i, n = n)

def _convert(res, columns):
    """
    We run a result per each column, now we want to convert it back to objects
    ----------
    res : dict
        results run per each column.
    """
    values = list(res.values())
    if is_tss(values):
        return pd.DataFrame(res)
    elif is_arrs(values) and is_int(columns):
        return np.array(values).T
    elif is_tuples(values):
        return tuple([_convert(dict(zip(res.keys(), row)), columns) for row in zipper(*values)])
    else:    
        return np.array(values) if is_int(columns) else pd.Series(res)

def df_sync(dfs, join = 'ij', method = None, columns = 'ij'):
    """
    df_sync performs a sync of multiple dataframes
    
    :Parameters:
    ----------
    dfs : list or dict of timeseries
        dataframes to be synched
    join : str, optional
        index join method. The default is 'ij'.
    method : str/float, optional
        how the nan's are to be filled once reindexing occurs. The default is None.
    columns : str, optional
        how to sync multi-column timeseries. The default is 'ij'.

    :Example:
    -------
    >>> a = pd.DataFrame(np.random.normal(0,1,(100,5)), drange(-100,-1), list('abcde'))
    >>> b = pd.DataFrame(np.random.normal(0,1,(100,5)), drange(-99), list('bcdef'))
    >>> c = 'not a timeseries'
    >>> d = pd.DataFrame(np.random.normal(0,1,(100,1)), drange(-98,1), ['single_column_df'])
    >>> s = pd.Series(np.random.normal(0,1,105), drange(-104))
    
    :Example: inner join on index and columns
    --------------------------------
    >>> dfs = [a,b,c,d,s]
    >>> join = 'ij'; method = None; columns = 'ij'
    >>> res = df_sync(dfs, 'ij')
    >>> assert len(res[0]) == len(res[1]) == len(res[-1]) == 98
    >>> assert res[2] == 'not a timeseries'
    >>> assert list(res[0].columns) == list('bcde')

    :Example: outer join on index and inner join on columns
    --------------------------------
    >>> res = df_sync(dfs, join = 'oj')
    >>> assert len(res[0]) == len(res[1]) == len(res[-1]) == 106; assert res[2] == 'not a timeseries'
    >>> assert list(res[0].columns) == list('bcde')

    >>> res = df_sync(dfs, join = 'oj', method = 1)
    >>> assert res[0].iloc[0].sum() == 4

    :Example: outer join on index and columns
    -------------------------------------------
    >>> res = df_sync(dfs, join = 'oj', method = 1, columns = 'oj')
    >>> assert res[0].iloc[0].sum() == 5
    >>> assert list(res[0].columns) == list('abcdef')
    >>> assert list(res[-2].columns) == ['single_column_df'] # single column unaffected

    :Example: synching of dict rather than a list
    -------------------------------------------
    >>> dfs = Dict(a = a, b = b, c = c, d = d, s = s)
    >>> res = df_sync(dfs, join = 'oj', method = 1, columns = 'oj')
    >>> assert res.c == 'not a timeseries'
    >>> assert res.a.shape == (106,6)
    """
    if isinstance(dfs, dict):
        values = list(dfs.values())
    elif isinstance(dfs, (list, tuple)):
        values = list(dfs)
    else:
        return dfs
    listed = _list(values)
    tss = [ts for ts in listed if is_df(ts)]
    index = df_index(listed, join)
    dfs = df_reindex(dfs, index, method = method)

    ### now we do the columns
    if columns is False or columns is None:
        return dfs
    else:
        cols = df_columns(tss, columns)
        dfs = df_recolumn(dfs, cols)
    return dfs

def _mask(df, value):
    if is_num(value) and np.isnan(value):
        return np.isnan(df)
    elif is_num(value):
        return df == value
    elif isinstance(value, list):
        return reducer(np.maximum, [_mask(df, v) for v in value]) 
    else:
        raise ValueError('not sure how to construct a mask from %s'%value)

def df_apply(df, func, axis = 0, raw = False, result_type = None, args = None, params = None, exc = np.nan):
    """
    similar to df.apply(func, axis...) except we nan results when we don't have any entries in original calculation

    :Example:
    ---------
    >>> df = pd.DataFrame(dict(a = [np.nan, np.nan], b = [np.nan, 1.]))
    >>> assert np.isnan(df_apply(df, 'sum').a)
    >>> assert df.apply('sum').a == 0
    
    """
    params = params or {}
    args = args or ()
    res = df.apply(func, axis = axis, raw = False, result_type = None, args = args, **params)
    if len(as_list(exc)) > 0:
        mask = ~_mask(df, exc)
        res[mask.sum(axis = axis) == 0] = np.nan
    return res   

class presync(wrapper):
    """
    Much of timeseries analysis in Pandas is spent aligning multiple timeseries before feeding them into a function.
    presync allows easy presynching of all paramters of a function.
    
    :Parameters:
    ----------
    function : callable, optional
        function to be presynched. The default is None.
    index : str, optional
        index join policy. The default is 'inner'.
    method : str/int/list of these, optional
        method of nan handling. The default is None.
    columns : str, optional
        columns join policy. The default is 'inner'.
    default : float, optional
        value when no data is available. The default is np.nan.

    :Returns:
    -------
    presynch-decorated function

    
    :Example:
    -------    
    >>> from pyg import *
    >>> x = pd.Series([1,2,3,4], drange(-3))
    >>> y = pd.Series([1,2,3,4], drange(-4,-1))    
    >>> z = pd.DataFrame([[1,2],[3,4]], drange(-3,-2), ['a','b'])
    >>> addition = lambda a, b: a+b    

    #We get some nonsensical results:

    >>> assert list(addition(x,z).columns) ==  list(x.index) + ['a', 'b']
    
    #But:
        
    >>> assert list(presync(addition)(x,z).columns) == ['a', 'b']
    >>> res = presync(addition, index='outer', method = 'ffill')(x,z)
    >>> assert eq(res.a.values, np.array([2,5,6,7]))
    
    
    :Example 2: alignment works for parameters 'buried' within...
    -------------------------------------------------------
    >>> function = lambda a, b: a['x'] + a['y'] + b    
    >>> f = presync(function, 'outer', method = 'ffill')
    >>> res = f(dict(x = x, y = y), b = z)
    >>> assert eq(res, pd.DataFrame(dict(a = [np.nan, 4, 8, 10, 11], b = [np.nan, 5, 9, 11, 12]), index = drange(-4)))
    
    
    :Example 3: alignment of numpy arrays
    -------------------------------------
    >>> addition = lambda a, b: a+b
    >>> a = presync(addition)
    >>> assert eq(a(pd.Series([1,2,3,4], drange(-3)), np.array([[1,2,3,4]]).T),  pd.Series([2,4,6,8], drange(-3)))
    >>> assert eq(a(pd.Series([1,2,3,4], drange(-3)), np.array([1,2,3,4])),  pd.Series([2,4,6,8], drange(-3)))
    >>> assert eq(a(pd.Series([1,2,3,4], drange(-3)), np.array([[1,2,3,4],[5,6,7,8]]).T),  pd.DataFrame({0:[2,4,6,8], 1:[6,8,10,12]}, drange(-3)))
    >>> assert eq(a(np.array([1,2,3,4]), np.array([[1,2,3,4]]).T),  np.array([2,4,6,8]))


    :Example 4: inner join alignment of columns in dataframes by default
    ---------------------------------------------------------------------
    >>> x = pd.DataFrame({'a':[2,4,6,8], 'b':[6,8,10,12.]}, drange(-3))
    >>> y = pd.DataFrame({'wrong':[2,4,6,8], 'columns':[6,8,10,12]}, drange(-3))
    >>> assert len(a(x,y)) == 0    
    >>> y = pd.DataFrame({'a':[2,4,6,8], 'other':[6,8,10,12.]}, drange(-3))
    >>> assert eq(a(x,y),x[['a']]*2)
    >>> y = pd.DataFrame({'a':[2,4,6,8], 'b':[6,8,10,12.]}, drange(-3))
    >>> assert eq(a(x,y),x*2)
    >>> y = pd.DataFrame({'column name for a single column dataframe is ignored':[1,1,1,1]}, drange(-3)) 
    >>> assert eq(a(x,y),x+1)
    
    >>> a = presync(addition, columns = 'outer')
    >>> y = pd.DataFrame({'other':[2,4,6,8], 'a':[6,8,10,12]}, drange(-3))
    >>> assert sorted(a(x,y).columns) == ['a','b','other']    

    :Example 4: ffilling, bfilling
    ------------------------------
    >>> x = pd.Series([1.,np.nan,3.,4.], drange(-3))    
    >>> y = pd.Series([1.,np.nan,3.,4.], drange(-4,-1))    
    >>> assert eq(a(x,y), pd.Series([np.nan, np.nan,7], drange(-3,-1)))

    but, we provide easy conversion of internal parameters of presync:

    >>> assert eq(a.ffill(x,y), pd.Series([2,4,7], drange(-3,-1)))
    >>> assert eq(a.bfill(x,y), pd.Series([4,6,7], drange(-3,-1)))
    >>> assert eq(a.oj(x,y), pd.Series([np.nan, np.nan, np.nan, 7, np.nan], drange(-4)))
    >>> assert eq(a.oj.ffill(x,y), pd.Series([np.nan, 2, 4, 7, 8], drange(-4)))
    
    :Example 5: indexing to a specific index
    ----------------------------------------
    >>> index = pd.Index([dt(-3), dt(-1)])
    >>> a = presync(addition, index = index)
    >>> x = pd.Series([1.,np.nan,3.,4.], drange(-3))    
    >>> y = pd.Series([1.,np.nan,3.,4.], drange(-4,-1))    
    >>> assert eq(a(x,y), pd.Series([np.nan, 7], index))
    
    
    :Example 6: returning complicated stuff
    ----------------------------------------
    >>> from pyg import * 
    >>> a = pd.DataFrame(np.random.normal(0,1,(100,10)), drange(-99))
    >>> b = pd.DataFrame(np.random.normal(0,1,(100,10)), drange(-99))

    >>> def f(a, b):
    >>>     return (a*b, ts_sum(a), ts_sum(b))

    >>> old = f(a,b)    
    >>> self = presync(f)
    >>> args = (); kwargs = dict(a = a, b = b)
    >>> new = self(*args, **kwargs)
    >>> assert eq(new, old)
    """
    
    def __init__(self, function = None, index = 'inner', method = None, columns = 'inner', default = np.nan):
        super(presync, self).__init__(function = function, index = index, method = method, columns = columns , default = default)
    
    @property
    def ij(self):
        return copy(self) + dict(index = 'inner')

    @property
    def oj(self):
        return self + dict(index = 'outer')

    @property
    def lj(self):
        return self + dict(index = 'left')

    @property
    def rj(self):
        return self + dict(index = 'right')

    @property
    def ffill(self):
        return copy(self) + dict(method = 'ffill')

    @property
    def bfill(self):
        return self + dict(method = 'bfill')


    def wrapped(self, *args, **kwargs):
        _idx = kwargs.pop('join', self.index)
        _method = kwargs.pop('method', self.method)
        _columns = kwargs.pop('columns', self.columns)
        
        values = list(args) + list(kwargs.values())
        listed = _list(values)
        tss = [ts for ts in listed if is_ts(ts)]
        callargs = inspect.getcallargs(self.function, *args, **kwargs)
        if is_str(_idx) and _idx in callargs:
            index = _index(callargs[_idx])
        else:
            index = df_index(listed, _idx)
        args_= df_reindex(args, index, method = _method)
        kwargs_= df_reindex(kwargs, index, method = _method)
        ### now we do the columns
        if _columns is False:
            return self.function(*args_, **kwargs_)
        else:
            cols = [tuple(ts.columns) for ts in tss if is_df(ts) and ts.shape[1]>1]
            if len(set(cols))==1: # special case where all 2-d dataframes have same column headers
                columns = cols[0]
                n = len(columns)
                res = {column: self.function(*df_column(args_,column = column, i = i, n = n), **df_column(kwargs_, column=column, i = i, n = n)) for i, column in enumerate(columns)}
            else:
                columns = df_columns(listed, _columns)
                if is_int(columns):
                    res = {i: self.function(*df_column(args_, column = None, i = i), **df_column(kwargs_, column=None, i = i)) for i in range(columns)}
                elif columns is None:
                    res = self.function(*df_column(args_, column = None), **df_column(kwargs_, column = None))
                    if isinstance(res, pd.Series):
                        cols1 = [tuple(ts.columns) for ts in tss if is_df(ts) and ts.shape[1]==1]
                        if len(set(cols1))>0:
                            res = pd.DataFrame(res) 
                    return res
                else:
                    columns = list(columns) if isinstance(columns, pd.Index) else as_list(columns)
                    columns = sorted(columns)
                    res = {column: self.function(*df_column(args_,column = column), **df_column(kwargs_, column=column)) for column in columns}                
            converted = _convert(res, columns)
            return converted 


@presync
def _div_(a, b):
    """
    division of a by b supporting presynching (inner join) of timeseries
    """
    if is_num(b):
        return np.nan if b == 0 else a/b
    else:
        denom = b.copy()
        denom[denom == 0] = np.nan
        return a/denom

@presync
def _sub_(a, b):
    """
    subtraction of b from a supporting presynching (inner join) of timeseries
    """
    return a-b

@presync
def _add_(a, b):
    """
    addition of a and b supporting presynching (inner join) of timeseries
    """
    return a + b

@presync
def _mul_(a, b):
    """
    multiplication of b and a supporting presynching (inner join) of timeseries
    """
    return a * b

@presync
def _pow_(a, b):
    """
    equivalent to a**b supporting presynching (inner join) of timeseries
    """
    return a**b


@presync
def _gt_(a, b):
    """
    comparison of a and b supporting presynching (inner join) of timeseries
    """
    return a > b

@presync
def _ge_(a, b):
    """
    comparison of a and b supporting presynching (inner join) of timeseries
    """
    return a >= b


@presync
def _lt_(a, b):
    """
    comparison of a and b supporting presynching (inner join) of timeseries
    """
    return a < b

@presync
def _le_(a, b):
    """
    comparison of a and b supporting presynching (inner join) of timeseries
    """
    return a <= b



def add_(a, b = None, join = 'ij', method = None, columns = 'ij'):
    """
    a = pd.Series([1,2,3], drange(-2))
    b = pd.Series([1,2,3], drange(-3,-1))
    add_(a,b, 'oj', method = 0)
    
    addition of a and b supporting presynching (inner join) of timeseries
    """
    dfs = as_list(a) + as_list(b)
    f = lambda a, b: _add_(a, b, join = join, method = method, columns = columns)
    return reducer(f, dfs)
    

def mul_(a, b = None, join = 'ij', method = None, columns = 'ij'):
    """
    multiplication of a and b supporting presynching (inner join) of timeseries
    mul_(a,b,join = 'oj', method = 'ffill')
    cell(mul_, a = a, b = b, join = 'oj')()
    """
    dfs = as_list(a) + as_list(b)
    f = lambda a, b: _mul_(a, b, join = join, method = method, columns = columns)
    return reducer(f, dfs)

def div_(a, b, join = 'ij', method = None, columns = 'ij'):
    """
    division of a by b supporting presynching (inner join) of timeseries
    """
    if isinstance(a, list):
        a = mul_(a, join = join, method = method, columns = columns)
    if isinstance(b, list):
        b = mul_(b, join = join, method = method, columns = columns)
    return _div_(a, b, join = join, method = method, columns = columns)

def sub_(a, b, join = 'ij', method = None, columns = 'ij'):
    """
    subtraction of b from a supporting presynching (inner join) of timeseries
    """
    if isinstance(a, list):
        a = add_(a, join = join, method = method, columns = columns)
    if isinstance(b, list):
        b = add_(b, join = join, method = method, columns = columns)
    return _sub_(a, b, join = join, method = method, columns = columns)

def pow_(a, b, join = 'ij', method = None, columns = 'ij'):
    """
    equivalent to a**b supporting presynching (inner join) of timeseries
    """
    return _pow_(a,b, join = join, method = method, columns = columns)

def gt_(a, b, join = 'ij', method = None, columns = 'ij'):
    """
    equivalent to a>b supporting presynching (inner join) of timeseries
    """
    return _gt_(a,b, join = join, method = method, columns = columns)

def ge_(a, b, join = 'ij', method = None, columns = 'ij'):
    """
    equivalent to a>=b supporting presynching (inner join) of timeseries
    """
    return _ge_(a,b, join = join, method = method, columns = columns)

def lt_(a, b, join = 'ij', method = None, columns = 'ij'):
    """
    equivalent to a<b supporting presynching (inner join) of timeseries
    """
    return _lt_(a,b, join = join, method = method, columns = columns)

def le_(a, b, join = 'ij', method = None, columns = 'ij'):
    """
    equivalent to a<=b supporting presynching (inner join) of timeseries
    """
    return _le_(a,b, join = join, method = method, columns = columns)

def _align_columns(a, b, func):
    da, db = [len(getattr(x, 'shape', ())) for x in (a,b)]
    if min(da, db) == 0:
        return func(a,b)
    if da == 2 and db == 2 and list(a.columns) == list(b.columns): # perfect fit, no need to run as_series
        return func(a,b)        
    a,b = as_series(a), as_series(b)
    da, db = [len(getattr(x, 'shape', ())) for x in (a,b)]
    if da == db:       # (1,1) and (2,2)
        return func(a,b)
    if is_series(b):   # (2,1)
        b = pd.concat([b] * a.shape[1], axis=1)
        b.columns = a.columns
    elif is_series(a): # (1,2)
        a = pd.concat([a] * b.shape[1], axis=1)
        a.columns = b.columns
    return func(a,b)
        
def _minimum(a, b):
    return _align_columns(a, b, np.minimum)

def _maximum(a, b):
    return _align_columns(a, b, np.maximum)


def min_(a, b = None, join = 'ij', method = None, columns = 'ij'):
    """
    equivalent to redced np.minimum operation supporting presynching of timeseries
    """
    dfs = as_list(a) + as_list(b)
    dfs = df_sync(dfs, join = join, method = method, columns = columns)
    return reducer(_minimum, dfs)

def max_(a, b = None, join = 'ij', method = None, columns = 'ij'):
    """
    equivalent to redced np.minimum operation supporting presynching of timeseries
    """
    dfs = as_list(a) + as_list(b)
    dfs = df_sync(dfs, join = join, method = method, columns = columns)
    return reducer(_maximum, dfs)

def mask2v(df, mask = np.nan, value = 0.0):
    if not is_pd(mask):
        mask = _mask(df, mask)
    res = df.copy()
    res[mask] = value
    return res

def df_count(a, b = None, join = 'oj', method = None, columns = 'oj', exc = np.nan):
    """
    apply count() on multiple dataframes
    - synch the data
    - removes nans (or any value in mask)

    :Example:
    ----------
    >>> a = pd.DataFrame(dict(a = [1, 2], b = [np.nan, np.nan], c = [1, 2.]))
    >>> b = pd.DataFrame(dict(a = [np.nan,2], b = [2, np.nan]))
    >>> df_count(a,b, columns = 'oj')
    
       a  b  c
    0  1  1  1
    1  2  0  1
    
    """
    dfs = as_list(a) + as_list(b)
    dfs = df_sync(dfs, join = join, method = method, columns = columns)
    masks = [_mask(df, exc) for df in dfs]
    n = sum([~mask for mask in masks])
    return n


def df_sum(a, b = None, join = 'oj', method = None, columns = 'oj', exc = np.nan):
    """
    apply sum() on multiple dataframes
    - synch the data
    - removes nans (or any value in mask)
    
    :Example:
    ----------
    >>> a = pd.DataFrame(dict(a = [1, 2], b = [np.nan, np.nan], c = [1, 2.]))
    >>> b = pd.DataFrame(dict(a = [np.nan,2], b = [2, np.nan]))
    >>> df_sum(a,b)
    
         a    b    c
    0  1.0  2.0  1.0
    1  4.0  NaN  2.0
    
    """
    dfs = as_list(a) + as_list(b)
    dfs = df_sync(dfs, join = join, method = method, columns = columns)
    masks = [_mask(df, exc) for df in dfs]
    clean_dfs = [mask2v(df, mask, 0.0) for df, mask in zip(dfs, masks)]
    res = sum(clean_dfs)
    n = sum([~mask for mask in masks])
    res[n == 0] = np.nan
    return res       
        
def df_mean(a, b = None, join = 'oj', method = None, columns = 'oj', exc = np.nan):
    """
    apply mean() on multiple dataframes
    - synch the data
    - removes nans (or any value in mask)
    
    :Example:
    ----------
    >>> a = pd.DataFrame(dict(a = [1, 2], b = [np.nan, np.nan], c = [1, 2.]))
    >>> b = pd.DataFrame(dict(a = [np.nan,2], b = [2, np.nan]))
    >>> df_mean(a,b)
    
         a    b    c
    0  1.0  2.0  1.0
    1  2.0  NaN  2.0

    """
    dfs = as_list(a) + as_list(b)
    dfs = df_sync(dfs, join = join, method = method, columns = columns)
    masks = [_mask(df, exc) for df in dfs]
    clean_dfs = [mask2v(df, mask, 0.0) for df, mask in zip(dfs, masks)]
    res = sum(clean_dfs)
    n = sum([~mask for mask in masks])
    n[n == 0] = np.nan
    return res / n

def df_std(a, b = None, join = 'oj', method = None, columns = 'oj', exc = np.nan):
    """
    apply biased std() on multiple dataframes
    - synch the data
    - removes nans (or any value in mask)
    
    :Example:
    ----------
    >>> a = pd.DataFrame(dict(a = [1, 2], b = [np.nan, np.nan], c = [1, 2.]))
    >>> b = pd.DataFrame(dict(a = [np.nan,2], b = [2, np.nan]))
    >>> df_std(a,b)
    
         a    b    c
    0  1.0  2.0  1.0
    1  2.0  NaN  2.0

    """
    dfs = as_list(a) + as_list(b)
    dfs = df_sync(dfs, join = join, method = method, columns = columns)
    masks = [_mask(df, exc) for df in dfs]
    clean_dfs = [mask2v(df, mask, 0.0) for df, mask in zip(dfs, masks)]
    m1 = sum(clean_dfs)
    m2 = sum([df**2 for df in clean_dfs])
    n = sum([~mask for mask in masks])
    n[n < 2] = np.nan
    return (m2/n - (m1/n)**2) ** 0.5




def _closed(oc):
    if oc in '()oO':
        return False
    elif oc in '[]cC':
        return True
    else:
        raise ValueError('not sure how to parse boundary %s'%oc)
    

def _df_slice(df, lb = None, ub = None, openclose = '[)'):
    """    
    Performs a one-time slice of the dataframe. Does not stich slices together
    
    pandas slices has two issues:
        1) it fails for timeseries quite a but
        2) for timeseries df[dt1:dt2] is close-close while for normal dataframe df[lb,ub] is close-open
    
    """
    if isinstance(df, (pd.Index, pd.Series, pd.DataFrame)) and len(df)>0 and (ub is not None or lb is not None):
        l,u = openclose if openclose else '[)'
        l = _closed(l); u = _closed(u)
        if is_ts(df):
            lb = lb if lb is None or isinstance(lb, datetime.time) else dt(lb)
            ub = ub if ub is None or isinstance(ub, datetime.time) else dt(ub)
            if (l or lb is None) and (u or ub is None):
                try:
                    return df[lb:ub]
                except Exception:
                    pass
        elif (l or lb is None) and (ub is None or not u):
            try:
                return df[lb:ub]
            except Exception:
                pass
        if lb is not None:
            index = df if isinstance(df, pd.Index) else df.index
            if isinstance(lb, datetime.time):
                index = index.time
            df = df[index>=lb] if l else df[index>lb]
        if ub is not None:
            index = df if isinstance(df, pd.Index) else df.index
            if isinstance(ub, datetime.time):
                index = index.time            
            df = df[index<=ub] if u else df[index<ub]
    return df


def df_slice(df, lb = None, ub = None, openclose = '(]', n = 1):
    """
    slices a dataframe/series/index based on lower/upper bounds.
    If multiple timeseries are sliced at different times, will then stitch them together.
    
    :Parameters:
    ----------
    df : dataframe
        Either a single dataframe or a list of dataframes.
    lb : single or multiple lower bounds
        lower bounds to cut the data.
    ub : single or multiple upper bounds
        upper bounds to cut the data
    openclose : 2-character string
        defines how left/right boundary behave.
        [,] or c : close
        (,) or o : open
        ' ' : do not cut
    n : int
        This parameter is used when multiple timeseries are provided. 
        n determines the number of columns stitched together.
    
    :Returns:
    -------
    filtered (and possibly stictched) timeseries
    

    :Example: single timeseries filtering
    ---------
    >>> df = pd.Series(np.random.normal(0,1,1000), drange(-999))
    >>> df_slice(df, None, '-1m')
    >>> df_slice(df, '-1m', None)

    :Example: single timeseries, multiple filtering
    ---------
    >>> df = pd.Series(np.random.normal(0,1,1000), drange(-999))
    >>> lb = jan1 = drange(2018, None, '1y')
    >>> ub = feb1 = drange(dt(2018,2,1), None, '1y')
    >>> assert set(df_slice(df, jan1, feb1).index.month) == {1}


    :Example: single timeseries time of day filtering
    ---------
    >>> dates = drange(-5, 0, '5n')
    >>> df = pd.Series(np.random.normal(0,1,12*24*5+1), dates)
    >>> assert len(df_slice(df, None, datetime.time(hour = 10))) == 606
    >>> assert len(df_slice(df, datetime.time(hour = 5), datetime.time(hour = 10))) == 300
    >>> assert len(df_slice(df, lb = datetime.time(hour = 10), ub = datetime.time(hour = 5))) == len(dates) - 300


    :Example: stitching together multiple future contracts for a continuous price
    ---------
    >>> ub = drange(1980, 2000, '3m')
    >>> df = [pd.Series(np.random.normal(0,1,1000), drange(-999, date)) for date in ub]
    >>> df_slice(df, ub = ub)

    :Example: stitching together multiple future contracts for a continuous price in front 5 contracts
    ---------
    >>> ub = drange(1980, 2000, '3m')
    >>> df = [pd.Series(np.random.normal(0,1,1000), drange(-999, date)) for date in ub]
    >>> df_slice(df, ub = ub, n = 5).iloc[500:]

    :Example: stitching together symbols
    ---------
    >>> from pyg import * 
    >>> ub = drange(1980, 2000, '3m')
    >>> df = loop(list)(dt2str)(ub)
    >>> df_slice(df, ub = ub, n = 3)

    
    """
    if isinstance(lb, tuple) and len(lb) == 2 and ub is None:
        lb, ub = lb
    if isinstance(ub, datetime.time) and isinstance(lb, datetime.time) and lb>ub:
        pre  = df_slice(df, None, ub)
        post = df_slice(df, lb, None)
        return pd.concat([pre, post]).sort_index()        
    if isinstance(df, list): 
        if isinstance(lb, list) and ub is None:
            lb2df = dict(zip(lb,df))
            lb = sorted(lb); df = [lb2df[key] for key in lb]
            ub = lb[1:] + [None]
        elif isinstance(ub, list) and lb is None:
            ub2df = dict(zip(ub,df))
            ub = sorted(ub); df = [ub2df[key] for key in ub]
            lb = [None] + ub[:-1]
        boundaries = sorted(set([date for date in lb + ub if date is not None]))
        df = [d if is_pd(d) else pd.Series(d, boundaries) for d in df]
        if n > 1:
            df = [pd.concat(df[i: i+n], axis = 1) for i in range(len(df))]
            for d in df:
                d.columns = range(d.shape[1])
    dfs = as_list(df)
    dlu = zipper(dfs, lb, ub)
    res = [_df_slice(d, lb = l, ub = u, openclose = openclose) for d, l, u in dlu]
    if len(res) == 0:
        return None
    elif len(res) == 1:
        return res[0]
    elif isinstance(lb, list) and isinstance(ub, list):
        res = pd.concat(res)
    return res

def df_unslice(df, ub):
    """
    If we have a rolled multi-column timeseries, and we want to know where each timeseries is originally associated with.
    As long as you provide the stiching points, forming the upper bound of each original timeseries, 
    df_unslice will return a dict from each upper bound to a single-column timeseries

    :Example:
    ---------
    >>> ub = drange(1980, 2000, '3m')
    >>> dfs = [pd.Series(date.year * 100 + date.month, drange(-999, date)) for date in ub]
    >>> df = df_slice(dfs, ub = ub, n = 10)

    >>> df.iloc[700:-700:]    
    
    >>>                    0         1         2         3         4         5         6         7   8   9
    >>> 1979-03-08  198001.0  198004.0  198007.0  198010.0  198101.0  198104.0  198107.0  198110.0 NaN NaN
    >>> 1979-03-09  198001.0  198004.0  198007.0  198010.0  198101.0  198104.0  198107.0  198110.0 NaN NaN
    >>> 1979-03-10  198001.0  198004.0  198007.0  198010.0  198101.0  198104.0  198107.0  198110.0 NaN NaN
    >>> 1979-03-11  198001.0  198004.0  198007.0  198010.0  198101.0  198104.0  198107.0  198110.0 NaN NaN
    >>> 1979-03-12  198001.0  198004.0  198007.0  198010.0  198101.0  198104.0  198107.0  198110.0 NaN NaN
    >>>              ...       ...       ...       ...       ...       ...       ...       ...  ..  ..
    >>> 1998-01-27  199804.0  199807.0  199810.0  199901.0  199904.0  199907.0  199910.0  200001.0 NaN NaN
    >>> 1998-01-28  199804.0  199807.0  199810.0  199901.0  199904.0  199907.0  199910.0  200001.0 NaN NaN
    >>> 1998-01-29  199804.0  199807.0  199810.0  199901.0  199904.0  199907.0  199910.0  200001.0 NaN NaN
    >>> 1998-01-30  199804.0  199807.0  199810.0  199901.0  199904.0  199907.0  199910.0  200001.0 NaN NaN
    >>> 1998-01-31  199804.0  199807.0  199810.0  199901.0  199904.0  199907.0  199910.0  200001.0 NaN NaN

    >>> res = df_unslice(df, ub)
    >>> res[ub[0]]
    >>> 1977-04-07    198001.0
    >>> 1977-04-08    198001.0
    >>> 1977-04-09    198001.0
    >>> 1977-04-10    198001.0
    >>> 1977-04-11    198001.0
    >>>                 ...
    >>> 1979-12-28    198001.0
    >>> 1979-12-29    198001.0
    >>> 1979-12-30    198001.0
    >>> 1979-12-31    198001.0
    >>> 1980-01-01    198001.0
    >>> Name: 0, Length: 1000, dtype: float64
    
    We can then even slice the data again:
        
    >>> assert eq(df_slice(list(res.values()), ub = ub, n = 10), df)

    """
    n = df.shape[1] if is_df(df) else 1
    res = dictable(ub = ub, lb = [None] + ub[:-1], i = range(len(ub)))
    res = res(ts = lambda lb, ub: df_slice(df, lb, ub, '(]'))
    res = res(rs = lambda i, ts: dictable(u = ub[i: i+n], j = range(len(ub[i: i+n])))(ts = lambda j: ts[j]))
    rs = dictable.concat(res.rs).listby('u').do([pd.concat, nona], 'ts')
    return dict(rs['u', 'ts'])



def ts_gap(ts, gap = 'days', recent = True):
    """
    Calculate the gap going FORWARD in time. The last gap is measured against today
    
    Example
    --------
    >>> ts = pd.Series(0, drange(-100,-10, '1b'))
    >>> gaps = ts_gap(ts)
    >>> gaps
    
    2022-06-01     1
    2022-06-02     1
    2022-06-03     3
    2022-06-06     1
    2022-06-07     1
                  ..
    2022-08-24     1
    2022-08-25     1
    2022-08-26     3
    2022-08-29     1
    2022-08-30    10
    Length: 65, dtype: int64

    >>> assert gaps.iloc[-1] == 10
    
    """
    if len(ts) == 0:
        return ts
    if is_ts(ts):
        ts = ts.index
    if gap == 'days':
        days = [(t1-t0).days for t0, t1 in zip(ts[:-1], ts[1:])]
        now = dt(0)
        if recent:
            if ts[-1] <= now:
                days.append((dt(0) - ts[-1]).days)
            else:
                days.append(min(days))
            return pd.Series(days, index = ts)
        else:
            return pd.Series(days, index = ts[:-1])
    else:
        raise ValueError(f'pyg_base.ts_gap with gap={gap} not supported')

def ts_deal_with_issue(ts, issue_calc, issue_level = 0, deal = 'last'):
    """
    Detects an issue and then deal with it
    
    Parameters
    ----------
    
    
    """
    if not issue_level:
        return ts
    issue = issue_calc(ts)
    issues = issue[issue >= issue_level]
    if len(issues) == 0:
        return ts
    do = lower(deal)
    if is_str(do) and do == 'last': ## I just want to keep the last data which is good
        return df_slice(ts, issues.index[-1]) 
    elif is_str(do) and do == 'no_first': ## happy to keep 
        return df_slice(ts, issues.index[0]) 
    elif is_int(do): ## this is used to pick which section of the data you want, more general than 'last' and 'no_first'
        if len(issues) < max(-do, do+1):
            return ts
        else:
            return df_slice(ts, issues.index[do]) ## do = -1 === 'last', do = 0 == 'no_first' 
    elif is_str(do) and do == 'raise':
        raise ValueError(f'Found these issues {issues} at issue level >= {issue_level} with timeseries')
    else:
        raise ValueError(f'no idea how to deal with issues {issues} using {deal}')

def ts_degap(ts, max_gap = 0, deal = 'last', recent = False):
    """
    Parameters
    ----------
    ts : timeseries
        timeseries.
    max_gap : int, optional
        False, None or 0: don't do anything    
        Otherwise, anything over max_gap is defined as an issue. The default is 0, which actually means: do.
    deal : str, int, optional
        See 'ts_deal_with_issue' for range of options. The default is 'last', keeping the last segment of the data with no gaps

    Returns
    -------
    timeseries
        a timeseries with gap issues handled
    

    :Example: gaps in recent values do not matter by default
    ---------
    >>> from pyg import *
    >>> ts = pd.Series(1, drange(-1000,-500)) ## old data
    >>> assert eq(ts_degap(ts, 10), ts)
    >>> assert len(ts_degap(ts, 10, recent = True)) == 0 # There is no recent data
    """
    
    return ts_deal_with_issue(ts, issue_calc = partial(ts_gap, recent = recent), issue_level = max_gap, deal = deal)
   

def df_drop_index_duplicates(df, keep = 'last'):
    """
    drop duplicates based on index alone
    """
    ts = pd.Series(df.index)
    ts = ts.drop_duplicates(keep = keep)
    return df.iloc[ts.index]

    
