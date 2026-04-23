from pyg_base import interpolate, drange, eq, dt
import pandas as pd
import numpy as np


def test_interpolate_simple():
    a = np.array([1,2,3,4,5])
    res = interpolate(a, x = [0,4,8], y = [10,5, 2])
    assert eq(res, np.array([8.75, 7.5, 6.25, 5, 4.25]))    

def test_interpolate_ts():
    y = pd.DataFrame(np.array([
                        [1,np.nan, np.nan, 2.],
                       [4,5,6,7.],
                       [7,8,9,10]]), index = drange(-2))
    
    x = pd.DataFrame(np.array([
                       [1,2,3,4],
                       [4,5,6,7.],
                       [7,8,9,10]]), index = drange(-2))

    xnew = 7
    assert eq(interpolate(xnew, x = x, y = y), pd.Series([np.nan, 7, 7], drange(-2)))
    assert eq(np.round(interpolate(5, x = x, y = y, fill_value = 'extrapolate'),2), pd.Series([2.33, 5, 5.], drange(-2)))
