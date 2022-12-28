import pandas as pd
import numpy as np
import datetime
from pyg_base import shape, df_slice, drange, dt, dt_bump, dt2str, eq, dictable, df_unslice, nona, df_sync, Dict, add_, div_, mul_, sub_, pow_, df_fillna, df_reindex, min_, max_
from operator import add, itruediv, sub, mul, pow
from pyg_base._pandas import _align_columns

def test_df_slice():
    df = pd.Series(np.random.normal(0,1,1000), drange(-999, 2000))
    assert len(df_slice(df, None, dt_bump(2000,'-1m'))) == 969
    assert len(df_slice(df, dt_bump(2000,'-1m'), None)) == 31


    df = pd.Series(np.random.normal(0,1,1000), drange(-999, 2020))
    jan1 = drange(2018, dt(2021,11,1), '1y')
    feb1 = drange(dt(2018,2,1), dt(2021,11,1), '1y')
    res = df_slice(df, jan1, feb1, openclose = '[)')
    assert set(res.index.month) == {1}


def test_min_max_with_num():
    a = 2

    b = pd.Series([1,2,3])
    assert eq(min_(a, b), pd.Series([1,2,2]))
    assert eq(max_(a, b), pd.Series([2,2,3]))
    assert eq(min_(b, a), pd.Series([1,2,2]))
    assert eq(max_(b, a), pd.Series([2,2,3]))

    b = pd.DataFrame(dict(x=[1,2,3]))
    assert eq(min_(a, b), pd.DataFrame(dict(x = [1,2,2])))
    assert eq(max_(a, b), pd.DataFrame(dict(x = [2,2,3])))
    assert eq(min_(b, a), pd.DataFrame(dict(x = [1,2,2])))
    assert eq(max_(b, a), pd.DataFrame(dict(x = [2,2,3])))

    b = pd.DataFrame(dict(x=[1,2,3], y = [3,2,1]))
    assert eq(min_(a, b), pd.DataFrame(dict(x = [1,2,2], y = [2,2,1])))
    assert eq(max_(a, b), pd.DataFrame(dict(x = [2,2,3], y = [3,2,2])))
    assert eq(min_(b, a), pd.DataFrame(dict(x = [1,2,2], y = [2,2,1])))
    assert eq(max_(b, a), pd.DataFrame(dict(x = [2,2,3], y = [3,2,2])))

def test_min_max_with_series():
    a = pd.Series([2,2,2])
    b = pd.Series([1,2,3])
    assert eq(min_(a, b), pd.Series([1,2,2]))
    assert eq(max_(a, b), pd.Series([2,2,3]))
    assert eq(min_(b, a), pd.Series([1,2,2]))
    assert eq(max_(b, a), pd.Series([2,2,3]))

    b = pd.DataFrame(dict(x=[1,2,3]))
    assert eq(min_(a, b), pd.Series([1,2,2]))
    assert eq(max_(a, b), pd.Series([2,2,3]))
    assert eq(min_(b, a), pd.Series([1,2,2]))
    assert eq(max_(b, a), pd.Series([2,2,3]))

    b = pd.DataFrame(dict(x=[1,2,3], y = [3,2,1]))
    assert eq(min_(a, b), pd.DataFrame(dict(x = [1,2,2], y = [2,2,1])))
    assert eq(max_(a, b), pd.DataFrame(dict(x = [2,2,3], y = [3,2,2])))
    assert eq(min_(b, a), pd.DataFrame(dict(x = [1,2,2], y = [2,2,1])))
    assert eq(max_(b, a), pd.DataFrame(dict(x = [2,2,3], y = [3,2,2])))

def test_min_max_with_pseudoseries():
    a = pd.DataFrame(dict(a = [2,2,2]))
    b = pd.Series([1,2,3])
    assert eq(min_(a, b), pd.Series([1,2,2]))
    assert eq(max_(a, b), pd.Series([2,2,3]))
    assert eq(min_(b, a), pd.Series([1,2,2]))
    assert eq(max_(b, a), pd.Series([2,2,3]))

    b = pd.DataFrame(dict(x=[1,2,3]))
    assert eq(min_(a, b), pd.Series([1,2,2]))
    assert eq(max_(a, b), pd.Series([2,2,3]))
    assert eq(min_(b, a), pd.Series([1,2,2]))
    assert eq(max_(b, a), pd.Series([2,2,3]))

    b = pd.DataFrame(dict(a=[1,2,3]))
    assert eq(min_(a, b), pd.DataFrame(dict(a = [1,2,2])))
    assert eq(max_(a, b), pd.DataFrame(dict(a = [2,2,3])))
    assert eq(min_(b, a), pd.DataFrame(dict(a = [1,2,2])))
    assert eq(max_(b, a), pd.DataFrame(dict(a = [2,2,3])))

    b = pd.DataFrame(dict(x=[1,2,3], y = [3,2,1]))
    assert eq(min_(a, b), pd.DataFrame(dict(x = [1,2,2], y = [2,2,1])))
    assert eq(max_(a, b), pd.DataFrame(dict(x = [2,2,3], y = [3,2,2])))
    assert eq(min_(b, a), pd.DataFrame(dict(x = [1,2,2], y = [2,2,1])))
    assert eq(max_(b, a), pd.DataFrame(dict(x = [2,2,3], y = [3,2,2])))

def test_min_max_with_dataframe():
    a = pd.DataFrame(dict(a = [2,2,2], b = [2,2,2]))
    b = pd.DataFrame(dict(x = [1,2,3], y = [3,2,1]))
    assert eq(min_(a, b), pd.DataFrame(index=[0,1,2]))
    assert eq(max_(a, b), pd.DataFrame(index=[0,1,2]))
    a = pd.DataFrame(dict(x = [2,2,2], y = [2,2,2]))
    assert eq(min_(a, b), pd.DataFrame(dict(x = [1,2,2], y = [2,2,1])))
    assert eq(max_(a, b), pd.DataFrame(dict(x = [2,2,3], y = [3,2,2])))


def test_df_slice_time():
    dates = drange(-5, 2020, '5n')
    df = pd.Series(np.random.normal(0,1,12*24*5+1), dates)
    assert len(df_slice(df, None, datetime.time(hour = 10))) == 606
    assert len(df_slice(df, datetime.time(hour = 5), datetime.time(hour = 10))) == 300
    assert len(df_slice(df, lb = datetime.time(hour = 10), ub = datetime.time(hour = 5))) == len(dates) - 300


def test_df_slice_roll():
    ub = drange(1980, 2000, '3m')
    df = [pd.Series(np.random.normal(0,1,1000), drange(-999, date)) for date in ub]
    res = df_slice(df, ub = ub)
    assert len(res) == 8305
    ub = drange(1980, 2000, '3m')
    df = [pd.Series(np.random.normal(0,1,1000), drange(-999, date)) for date in ub]
    res = df_slice(df, ub = ub, n = 5).iloc[500:]
    res.shape == (7805,5)
# -*- coding: utf-8 -*-


def test_df_slice_roll_symbol():
    ub = drange(1980, 2000, '3m')
    df = [dt2str(date) for date in ub]    
    res = df_slice(df, ub = ub, n = 3)
    assert list(res.iloc[-3].values) == ['19990701', '19991001', '20000101']
    assert res.index[-3] == dt('19990701')
    res = df_slice(df, lb = ub, n = 3, openclose = '[)')
    assert list(res.iloc[-3].values) == ['19990701', '19991001', '20000101']
    assert res.index[-3] == dt('19990701')

def test_df_unslice():
    ub = drange(1980, 2000, '3m')
    dfs = [pd.Series(date.year * 100 + date.month, drange(-999, date)) for date in ub]
    df = df_slice(dfs, ub = ub, n = 10)
    res = df_unslice(df, ub)
    rs = dictable(res.items(), ['ub', 'df'])
    assert eq(df_slice(df = rs.df, ub = rs.ub, n = 10), df)
    assert len(rs.inc(lambda df: len(set(nona(df)))>1)) == 0

def test_df_sync():
    a = pd.DataFrame(np.random.normal(0,1,(100,5)), drange(-100,-1), list('abcde'))
    b = pd.DataFrame(np.random.normal(0,1,(100,5)), drange(-99), list('bcdef'))
    c = 'not a timeseries'
    d = pd.DataFrame(np.random.normal(0,1,(100,1)), drange(-98,1), ['single_column_df'])
    s = pd.Series(np.random.normal(0,1,105), drange(-104))
    
    dfs = [a,b,c,d,s]
    res = df_sync(dfs, 'ij')
    assert len(res[0]) == len(res[1]) == len(res[-1]) == 98
    assert res[2] == 'not a timeseries'
    assert list(res[0].columns) == list('bcde')

    res = df_sync(dfs, 'oj')
    assert len(res[0]) == len(res[1]) == len(res[-1]) == 106; 
    assert res[2] == 'not a timeseries'
    assert list(res[0].columns) == list('bcde')

    res = df_sync(dfs, join = 'oj', method = 1)
    assert res[0].iloc[0].sum() == 4

    res = df_sync(dfs, join = 'oj', method = 1, columns = 'oj')
    assert res[0].iloc[0].sum() == 5
    assert list(res[0].columns) == list('abcdef')
    assert list(res[-2].columns) == ['single_column_df'] # single column unaffected

    dfs = Dict(a = a, b = b, c = c, d = d, s = s)
    res = df_sync(dfs, join = 'oj', method = 1, columns = 'oj')
    assert res.c == 'not a timeseries'
    assert res.a.shape == (106,6)

def test_bi():
    s = pd.Series([1,2,3.], drange(-2,2000))
    a = pd.DataFrame(dict(a = [1,2,3.], b = [4,5,6.]), drange(-2,2000))
    b = pd.DataFrame(dict(c = [1,2,3.], b = [4,5,6.]), drange(-3,2000)[:-1])
    c = 5
    
    assert eq(add_(s,a), pd.DataFrame(dict(a = [2,4,6.], b = [5,7,9.]), drange(-2,2000)))
 
    for f in [add_,sub_,div_,mul_,pow_]:
        assert f(s,b).shape == (2,2)
        assert f(s,b, 'oj').shape == (4,2)

        assert f(a,b).shape == (2,1)
        assert f(a,b, 'oj').shape == (4,1)
        assert f(a,b, 'oj', 0, 'oj').shape == (4,3)

    for f in [add_,mul_]: # support for list of values
        for x in [s,a,b,c]:
            for y in [s, a, b, c]:
                assert eq(f([x,y]), f(x,y))
    
    assert add_([a,b,c,s], columns = 'oj').shape == (2,3)
    assert mul_([a,b,c,s], columns = 'oj').shape == (2,3)

    # operations with a constant
    for f,o in zip([add_,sub_,div_,mul_,pow_], [add, sub, itruediv,mul, pow]):
        for v in [a,b,s,c]:
            assert eq(f(v,c), o(v,c))


def test_df_fillna():
    from numpy import nan
    dates =  drange(dt(2000), dt(2000,1,14))
    df = pd.Series([0,nan, 1, nan, nan, 2, nan, nan, nan, 3, nan, nan, nan, nan, ], dates)
    df0 = df.copy()
    df0[np.isnan(df)] = 0
    assert eq(df_fillna(df), df)
    assert eq(df_fillna(df, 0), df0)
    assert eq(df_fillna(df, 'ffill_na'), pd.Series([0,0, 1, 1,1, 2, 2,2,2,3, nan, nan, nan, nan, ], dates))
    assert eq(df_fillna(df, 'ffill_0'), pd.Series([0,0, 1, 1,1, 2, 2,2,2,3] + [0] * 4, dates))
    assert eq(df_fillna(df, 'bfill'), pd.Series([0,1, 1, 2, 2, 2, 3, 3, 3, 3, nan, nan, nan, nan, ], dates))
    assert eq(df_fillna(df, 'linear'), pd.Series([0,.5, 1, 1+1/3, 1+2/3, 2, 2.25, 2.5, 2.75] + [3]*5, dates))
    

    
def test_sub_column_names_for_df():
    df = pd.DataFrame(range(10), index = drange(9), columns = ['df'])
    for f in (sub_, add_, div_, mul_):
        res = f(df,df)
        assert isinstance(res, pd.DataFrame)
        assert list(res.columns) == ['df']
    df2 = pd.DataFrame(range(10), index = drange(9), columns = ['df2'])
    for f in (sub_, add_, div_, mul_):
        res = f(df,df2)
        assert isinstance(res, pd.DataFrame)
        assert list(res.columns) == [0]
    

def test_align_columns():
    As = [1, 
          np.array([1.,1.]),
          np.array([1.,1.]).reshape((2,1)),
          np.array([[1.,1.],[1.,1.]]),
          pd.Series([1.,1.]),
          pd.DataFrame(dict(a = [1,1], b = [1,1]))
          ]
          
    Bs = As
    for a in As:
        for b in Bs:
            res = max_(a,b)
            assert len(shape(res)) == max(len(shape(a)), len(shape(b)))
            res = min_(a,b)
            assert len(shape(res)) == max(len(shape(a)), len(shape(b)))

def test_reindex_unindexable():
    for t in [1, 'not a pd', dt(4)]:
        assert df_reindex(t,  [dt(9), 'whatever']) == t
    