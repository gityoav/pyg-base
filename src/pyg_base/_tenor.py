import pandas as pd
from pyg_base._dates import dt
from pyg_base._types import is_date, is_ts, is_str
from pyg_base._pandas import df_reindex

def years_between(t0, t1):
    """
    whole years between two dates

    Example
    -------
    >>> t0 = dt(2000,1,1)
    >>> t1 = dt(2001,1,1)
    >>> assert years_between(t0,t1) == 1
    >>> assert years_between(dt(2000, 2, 1), t1) == 0

    
    """
    y = t1.year - t0.year
    if dt(t0.year + y, t0.month, t0.day) > t1:
        return y-1
    else:
        return y

_tenors = {'spot': 0, 'sn' : 1/365, 'tn' : 2/365, 's/n': 1/365, 't/n': 2/365}

def years_to_maturity(maturity, ts = None):
    """
    calculates years to maturity with part of the year calculated as ACT/365
    
    :Example:
    ---------
    >>> from pyg import *     
    >>> ts = pd.Series(range(1000), drange(-999))
    >>> maturity = dt('2Y')
    >>> years_to_maturity(maturity, ts)
    
    :Example: ts is None
    --------------------
    >>> assert years_to_maturity('1y') == 1
    >>> assert years_to_maturity('2Q') == 0.5
    >>> assert years_to_maturity('18m') == 1.5
    >>> assert years_to_maturity('13w') == 0.25
    >>> assert years_to_maturity('730d') == 2
    >>> assert years_to_maturity('spot') == 0    
    """
    if ts is None:
        if isinstance(maturity, (list, tuple)):
            return type(maturity)([years_to_maturity(m) for m in maturity])
        if is_str(maturity):
            maturity = maturity.lower()
            if maturity in _tenors:
                return _tenors[maturity]
            else:
                return int(maturity[:-1]) / dict(q = 4, y = 1, m = 12, w = 52, d = 365, b = 252)[maturity[-1]]
    if not is_date(maturity):
        return maturity
    if ts is None:
        ts = dt(0)
    if is_date(ts):
        y = years_between(ts, maturity)
        frac = (dt(maturity, f'-{y}y') - ts).days / 365.
        return y + frac
    elif isinstance(ts, list):
        return [years_to_maturity(maturity, t) for t in ts]             
    elif is_ts(ts):
        if len(ts) == 0:
            return ts
        t0 = ts.index[0]
        years = list(range(2+maturity.year-t0.year))[::-1]
        dates = [dt(maturity, f'-{y}y') for y in years]
        y = df_reindex(pd.Series(years, dates), ts, method = 'bfill')
        days = df_reindex(pd.Series(dates, dates), ts, method = 'bfill')
        frac = pd.Series((days.values - days.index).days / 365, ts.index)
        return y + frac
    else:
        raise ValueError(f'cannot calculate years_to_maturity for {maturity} and {ts}')
