from pyg_base import df_roll_off, dictable, dt, drange, eq
import numpy as np
import pandas as pd

def test_df_roll_off_handles_mixture_of_df_and_series_single_column():
    chain = dictable(dict(ts = [pd.Series(1, drange(-100)), pd.DataFrame(dict(a =2), drange(-100))], roll = [dt(-20), None]))
    res = df_roll_off(chain, loader = 'ts')['data']
    assert eq(res[dt(-19):] , pd.Series(2, drange(-19))) 
    assert eq(res[dt(-100):dt(-20)] , pd.Series(1, drange(-100,-20))) 

    ## now with old dataframe/pd.Series
    for data in [pd.Series(0, drange(-120, -80)), pd.DataFrame(dict(old = 0), drange(-120, -80))]:
        res = df_roll_off(chain, loader = 'ts', data = data)['data']
        assert len(res[res==0]) == 41
        assert eq(res[dt(-19):] , pd.Series(2, drange(-19))) 
        assert eq(res[dt(-79):dt(-20)] , pd.Series(1, drange(-79,-20))) 


def test_df_roll_off_handles_multiple_columns():
    chain = dictable(dict(ts = [pd.Series(1, drange(-100)), pd.DataFrame(dict(a =2), drange(-100))], roll = [dt(-20), None]))
    res = df_roll_off(chain, loader = 'ts',  n = 2)['data']
    assert eq(res[dt(-19):] , pd.DataFrame({0:2, 1: np.nan}, drange(-19))) 
    assert eq(res[dt(-100):dt(-20)] , pd.DataFrame({0:1,1:2}, drange(-100,-20))) 

    ## now with old dataframe/pd.Series
    for data in [pd.DataFrame({0:0,1:1}, drange(-120, -80)), pd.DataFrame({0:0,1:1,2:2}, drange(-120, -80))]:
        res = df_roll_off(chain, loader = 'ts', data = data, n = 2)['data']
        assert eq(res[:dt(-80)], data[[0,1]])
        assert eq(res[dt(-19):] , pd.DataFrame({0:2, 1: np.nan}, drange(-19)))
        assert eq(res[dt(-79):dt(-20)] , pd.DataFrame({0:1,1:2}, drange(-79,-20))) 
