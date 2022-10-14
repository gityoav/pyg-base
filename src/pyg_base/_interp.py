import scipy
import numpy as np
from scipy.interpolate import interp1d
from pyg_base._types import is_ts, is_num, is_arr, is_df, is_strs, is_nan
from pyg_base._pandas import df_reindex
from pyg_base._loop import pd2np

def _val(x, i, n):
    if is_num(x):
        return x
    if len(x.shape) < 2:
        return x
    elif len(x.shape) == 2:
        return x[i]
    

@pd2np
def _interpolate(y,x,xnew, 
                   kind = 'linear', 
                   axis = -1, 
                   copy = False, 
                   bounds_error = False,
                   fill_value = np.nan,
                   assume_sorted = True,
                   min_n = 2):
    """
    
    :Example: simple interplation with nan
    ---------

    >>> x = np.array([1,2,3,4])
    >>> y = np.array([1,np.nan, np.nan, 2.])
    >>> assert _interpolate(x = x, y = y, xnew = -2, fill_value = 'extrapolate') == 0

    :Example: simple interplation with y in 2d
    ---------
    >>> x = np.array([1,2,3,4])
    >>> y = np.array([[1,np.nan, np.nan, 2.],
                      [4,5,6,7.]])
    >>> assert eq(_interpolate(y, x, 2.5), np.array([1.5, 5.5]))
    >>> assert eq(_interpolate(y, x, [2.5,2.5]), np.array([1.5, 5.5]))
    >>> assert eq(_interpolate(y, np.array([x,x]), [2.5,2.5]), np.array([1.5, 5.5]))

    """

    if len(y.shape) == 2:
        if is_arr(x) and len(x.shape)==2:
            xs = x
        else:
            xs = [x] * y.shape[0]
        if isinstance(xnew, list):
            xnew = np.array(xnew)
        if is_arr(xnew) and len(xnew.shape) == 2:
            xnews = xnew
        elif is_arr(xnew) and len(xnew.shape) == 1 and len(xnew) == y.shape[0]:
            xnews = xnew
        else:
            xnews = [xnew] * y.shape[0]
        res = np.array([_interpolate(x = xs[i], y = y[i], xnew = xnews[i], 
                           kind = kind, 
                           axis = axis, 
                           copy = copy, 
                           bounds_error = bounds_error,
                           fill_value = fill_value,
                           assume_sorted = assume_sorted)
                           for i in range(y.shape[0])])
        return res
    if is_nan(xnew):
        return np.nan
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(y) < min_n:
        return np.nan + xnew
    return interp1d(x, y, kind = kind, axis = axis, copy = copy, 
                    bounds_error = bounds_error, 
                    fill_value = fill_value,
                    assume_sorted = True)(xnew)

def interpolate(a, y, x = None, 
                   kind = 'linear', 
                   axis = -1, 
                   copy = False, 
                   bounds_error = False,
                   fill_value = np.nan,
                   assume_sorted = True,
                   xmethod = 'linear'):
    """
    implements scipy.interpolate.interp1d with support:
        - handling nan for y values 
        - pandas objects
    
    :Example:
    ---------
    >>> y = pd.DataFrame(np.array([[1,np.nan, np.nan, 2.],
                      [4,5,6,7.]]), index = drange(-1))
    
    >>> x = [0,1,2,3]    
    >>> xnew = 1.5
    >>> assert eq(interpolate(x,y,xnew), pd.Series([1.5, 5.5], drange(-1)))

    :Example: interpolated x
    ---------
    >>> y = pd.DataFrame(np.array([
                        [1,np.nan, np.nan, 2.],
                       [4,5,6,7.],
                       [7,8,9,10]]), index = drange(-2))
    
    >>> x = pd.DataFrame(np.array([
                       [4,5,6,7.],
                       [7,8,9,10]]), index = [dt(-2),dt(0)])

    >>> xnew = 7
    >>> assert eq(interpolate(x,y,xnew), pd.Series([2., 5.5, 7.], drange(-2)))
    >>> assert eq(interpolate(x,y,5, fill_value = 'extrapolate'), pd.Series([4/3., 3.5, 5.], drange(-2)))
    
    """
    if x is None:
        x = y.columns.values
    elif isinstance(x, list):
        x = np.array(x)
    if isinstance(a, list):
        a = np.array(a)
    if is_ts(a):
        y = df_reindex(y, a) if is_ts(y) else y
        x = df_reindex(x, a, method = xmethod) if is_ts(x) else x
    if is_df(y):
        x = df_reindex(x, y, method = xmethod) if is_df(x) else x
    res = _interpolate(y = y,x = x,xnew = a, kind = kind, 
                           axis = axis, 
                           copy = copy, 
                           bounds_error = bounds_error,
                           fill_value = fill_value,
                           assume_sorted = assume_sorted)
    if is_df(res) and is_df(a) and res.shape[1] == a.shape[1]:
        res.columns = a.columns
    return res
