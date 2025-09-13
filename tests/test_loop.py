from pyg_base import loop, eq, drange, Dict, dictattr, pd2np, tree_loop, is_tree_deep
from pyg_base._loop import _item_by_key
import pandas as pd; import numpy as np
import pytest
from numpy import array
from collections import OrderedDict

SP = lambda a, b: Dict(s = a+b, p = a*b)
AB = lambda a, b: a+b

def S(v):
    if isinstance(v, list):
        return [S(w) for w in v]
    else:
        return v.s


def test_loop_dict():
    f = loop(dict)(AB)
    assert f(1,2) == 3
    assert f(1, b=2) == 3
    assert f(a=1, b=2) ==3
    assert f(dict(a=1,b=2), 2) == dict(a = 3, b = 4)
    assert f(dict(a=1,b=2), dict(a=2, b=3)) == dict(a = 3, b = 5)
    b = pd.Series([2,3], ['a','b'])
    assert f(dict(a=1,b=2), b) == dict(a = 3, b = 5)
    b = pd.DataFrame([[2,3],[3,4]], columns = ['a','b'])
    assert eq(f(dict(a=1,b=2), b), dict(a = 1 + b.a, b = 2 + b.b))
    
    with pytest.raises(TypeError):
        f(dict(a=1,b=2), dict(a=2, b=3, c=6))
    with pytest.raises(TypeError):
        f(dict(a=1,b=2), [2,2]) 
        
def test_loop_dict_with_more_params():
    self = loop(dict)(AB)
    assert self(dict(x=1,y=2), 2) == dict(x = 3, y = 4)
    assert self(dict(x=1,y=2), dict(x = 1, y = 2)) == dict(x = 2, y = 4)
    assert self(a = dict(x=1,y=2), b = dict(x = 1, y = 2, z = 3)) == dict(x = 2, y = 4)


def test_loop_dict_with_tree_and_dict():
    self = loop(dict)(AB)
    a = dict(eq = dict(x=1, y = 2), bonds = dict(z = 3)) ## some fancy structure
    b = dict(x = 3,y=4,z=5) ## data as a flat key-value
    assert self(a = a, b = b) == dict(eq = dict(x = 4, y =6), bonds = dict(z = 8))
    

def test__item_by_key():
    value = dict(x = 1, y = 2, z = 3)
    key = 'x'
    keys = ['x', 'y']
    i = None
    assert _item_by_key(value, key, keys, i) == 1
    assert _item_by_key(pd.Series(value), key, keys, i) == 1
    assert list(_item_by_key(pd.DataFrame([value]), key, keys, i).values) == [1]
    assert list(_item_by_key(pd.DataFrame(pd.Series(value)), key, keys, i).values) == [1]
    

def test_loop_just_dicts():
    f = loop(dict)(AB)
    res = f(dictattr(x = 1), b = 2)
    assert res == dictattr(x = 3) and type(res) == dictattr
    res = f(Dict(x = 1, y = 2), b = 2)
    assert res == Dict(x = 3, y = 4) and type(res) == Dict
    res = f(OrderedDict([('x',1), ('y',2),]), b = 3)
    assert res == OrderedDict([('x',4), ('y',5),]) and type(res) == OrderedDict

    class FakeDict(dict):
        pass
    with pytest.raises(TypeError):
        f(FakeDict(x = 1, y = 2), b = 2)
    f = loop(dict, FakeDict)(AB)
    res = f(FakeDict(x = 1, y = 2), b = 2)
    assert res == FakeDict(x = 3, y = 4) and type(res) == FakeDict


def test_loop_list():
    f = loop(list)(AB)
    assert f(1,2) == 3
    assert f(1, b=2) == 3
    assert f(a=1, b=2) ==3
    assert f([1,2], 2) == [3,4]
    assert f([1,2], [2,3]) == [3,5]
    assert f([1,2], (2,3)) == [3,5]
    assert f([1,2], np.array([2,3])) == [3,5]
    assert f([1,2], pd.Series([2,3])) == [3,5]
    b = pd.Series([2,3], drange(-1)) # a timeseries is considered a single object
    assert eq(f([1,2], b), [1+b, 2+b])
    b = pd.DataFrame([[2,3],[3,4]], columns = ['a','b'])
    assert eq(f([1,2], b), [1+b.a, 2+b.b])
    b = pd.DataFrame([[2,3,4],[3,4,5], [5,6,7]], columns = ['a','b','c'])
    assert eq(f([1,2], b), [1+b, 2+b])
    with pytest.raises(TypeError):
        f([1,2], [2,2,3]) 

    assert f((1,2),(3,4)) == (1,2,3,4)


def test_loop_list_dict():
    f = loop(list)(SP)
    assert f(1,2).s == 3
    assert f(1, b=2).s == 3
    assert f(a=1, b=2).s ==3
    assert S(f([1,2], 2)) == [3,4]
    assert S(f([1,2], [2,3])) == [3,5]
    assert S(f([1,2], (2,3))) == [3,5]
    assert S(f([1,2], np.array([2,3]))) == [3,5]
    assert S(f([1,2], pd.Series([2,3]))) == [3,5]
    b = pd.Series([2,3], drange(-1)) # a timeseries is considered a single object
    assert eq(S(f([1,2], b)), [1+b, 2+b])
    b = pd.DataFrame([[2,3],[3,4]], columns = ['a','b'])
    assert eq(S(f([1,2], b)), [1+b.a, 2+b.b])
    b = pd.DataFrame([[2,3,4],[3,4,5], [5,6,7]], columns = ['a','b','c'])
    assert eq(S(f([1,2], b)), [1+b, 2+b])
    with pytest.raises(TypeError):
        f([1,2], [2,2,3]) 



def test_loop_tuple():
    f = loop(tuple)(AB)
    assert f(1,2) == 3
    assert f(1, b=2) == 3
    assert f(a=1, b=2) ==3
    assert f((1,2), 2) == (3,4)
    assert f((1,2), [2,3]) == (3,5)
    assert f((1,2), np.array([2,3])) == (3,5)
    assert f((1,2), pd.Series([2,3])) == (3,5)
    b = pd.Series([2,3], drange(-1)) # a timeseries is considered a single object
    assert eq(f((1,2), b), (1+b, 2+b))
    b = pd.DataFrame([[2,3],[3,4]], columns = ['a','b'])
    assert eq(f((1,2), b), (1+b.a, 2+b.b))
    b = pd.DataFrame([[2,3,4],[3,4,5], [5,6,7]], columns = ['a','b','c'])
    assert eq(f((1,2), b), (1+b, 2+b))
    with pytest.raises(TypeError):
        f((1,2), [2,2,3]) 

    assert f([1,2],[3,4]) == [1,2,3,4]


def test_loop_series():
    f = loop(pd.Series)(AB)
    assert f(1,2) == 3
    assert f(1, b=2) == 3
    assert f(a=1, b=2) ==3
    a = pd.Series(dict(a=1,b=2))
    assert eq(f(a, 2) , a+2)
    assert eq(f(a,dict(a=2, b=3)) , pd.Series(dict(a = 3, b = 5)))
    ats = pd.Series([1,2], drange(-1))    
    assert eq(f(ats, 1) , ats+1)
    assert eq(list(f(a, [1,2]).values), [array([2, 3]), array([3, 4])])
    assert eq(list(f(a, array([1,2])).values), [array([2, 3]), array([3, 4])])
    f = loop(pd.Series)(np.std)
    assert eq(f(a) , pd.Series(dict(a=0, b=0)))
    assert f(ats) == 0.5
    
    
    
def test_loop_df():
    f = loop(pd.DataFrame)(AB)
    assert f(1,2) == 3
    assert f(1, b=2) == 3
    assert f(a=1, b=2) ==3
    a = pd.DataFrame(dict(a=[1,2],b=[2,3]))
    assert eq(f(a, 1), a + 1)
    assert eq(f(a, [1,2]), pd.DataFrame(dict(a=[2,3],b=[2+2,3+2])))
    assert eq(f(a, dict(a=1,b=2)), pd.DataFrame(dict(a=[2,3],b=[2+2,3+2])))
    ats = pd.DataFrame(dict(a=[1,2],b=[2,3]), drange(-1))
    f = loop(pd.Series)(np.std)
    assert eq(f(a), pd.Series(dict(a=0.5, b=0.5)))
    assert eq(f(ats), pd.Series(dict(a=0.5, b=0.5)))

    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]))
    b = np.array([1,2,3,4])
    f = loop(pd.DataFrame)(AB)
    assert eq(f(a,b).a.values, a.a.values + b)
    assert eq(f(a,b).b.values, a.b.values + b)

    b = np.array([[1,2,3,4],[0,-1,-2,-3]]).T
    assert eq(f(a,b).a.values, a.a.values + b.T[0])
    assert eq(f(a,b).b.values, a.b.values + b.T[1])

def test_loop_df_dict():
    f = loop(pd.DataFrame)(SP)
    assert f(1,2).s == 3
    assert f(1, b=2).s == 3
    assert f(a=1, b=2).s ==3
    a = pd.DataFrame(dict(a=[1,2],b=[2,3]))
    assert eq(f(a, 1).s, a + 1)
    assert eq(f(a, [1,2]).s, pd.DataFrame(dict(a=[2,3],b=[2+2,3+2])))
    assert eq(f(a, dict(a=1,b=2)).s, pd.DataFrame(dict(a=[2,3],b=[2+2,3+2])))
    ats = pd.DataFrame(dict(a=[1,2],b=[2,3]), drange(-1))
    f = loop(pd.Series)(lambda a: Dict(s = np.std(a)))
    assert eq(f(a).s, pd.Series(dict(a=0.5, b=0.5)))
    assert eq(f(ats).s, pd.Series(dict(a=0.5, b=0.5)))

    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]))
    b = np.array([1,2,3,4])
    f = loop(pd.DataFrame)(SP)
    assert eq(f(a,b).s.a.values, a.a.values + b)
    assert eq(f(a,b).s.b.values, a.b.values + b)

    b = np.array([[1,2,3,4],[0,-1,-2,-3]]).T
    assert eq(f(a,b).s.a.values, a.a.values + b.T[0])
    assert eq(f(a,b).s.b.values, a.b.values + b.T[1])


    

def test_loop_index_and_columns():
    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]), index = drange(3))
    b = np.array([1,2,3,4])
    f = loop(pd.DataFrame, pd.Series)(AB)
    assert eq(f(a,b).index, a.index)
    assert eq(f(a,b).columns, a.columns)
    s = pd.Series([1,2,3,4], drange(3))
    assert eq(f(s,b).index, s.index)

def test_loop_index_and_columns_dict():
    f = loop(pd.DataFrame, pd.Series)(SP)
    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]), index = drange(3))
    b = np.array([1,2,3,4])
    assert eq(f(a,b).s.index, a.index)
    assert eq(f(a,b).s.columns, a.columns)
    s = pd.Series([1,2,3,4], drange(3))
    assert eq(f(s,b).s.index, s.index)


    
def test_loop_df_axis1():
    f = loop(pd.DataFrame)(AB)
    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]))
    b = np.array([1,2])
    assert eq(f(a,b, axis=1).a.values, a.a.values + 1)
    assert eq(f(a,b, axis=1).b.values, a.b.values + 2)

def test_loop_df_axis1_dict():
    f = loop(pd.DataFrame)(SP)
    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]))
    b = np.array([1,2])
    assert eq(f(a,b, axis=1).s.a.values, a.a.values + 1)
    assert eq(f(a,b, axis=1).s.b.values, a.b.values + 2)



        
def test_loop_df_ndarray():
    f = loop(np.ndarray)(AB)
    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]))
    b = np.array([[1,2,3,4],[0,-1,-2,-3]]).T
    assert eq(f(b,a).T[0], a.a.values + b.T[0])
    assert eq(f(b,a).T[1], a.b.values + b.T[1])
    a = [1,2]        
    assert eq(f(b,a).T[0], b.T[0]+1)
    assert eq(f(b,a).T[1], b.T[1]+2)
    a = pd.DataFrame(dict(a=[1,2,3,4]))
    assert eq(f(b,a), np.array([[2,4,6,8],[1,1,1,1]]).T)
    a = a.values
    assert eq(f(b,a), np.array([[2,4,6,8],[1,1,1,1]]).T)

def test_loop_df_ndarray_dict():
    f = loop(np.ndarray)(SP)
    a = pd.DataFrame(dict(a=[1,2,3,4],b=[2,3,0,1]))
    b = np.array([[1,2,3,4],[0,-1,-2,-3]]).T
    assert eq(f(b,a).s.T[0], a.a.values + b.T[0])
    assert eq(f(b,a).s.T[1], a.b.values + b.T[1])
    a = [1,2]        
    assert eq(f(b,a).s.T[0], b.T[0]+1)
    assert eq(f(b,a).s.T[1], b.T[1]+2)
    a = pd.DataFrame(dict(a=[1,2,3,4]))
    assert eq(f(b,a).s, np.array([[2,4,6,8],[1,1,1,1]]).T)
    a = a.values
    assert eq(f(b,a).s, np.array([[2,4,6,8],[1,1,1,1]]).T)
    
 

def test_pd2np_exc():
    assert pd2np(exc = 'a')(lambda x, a: type(a))(x = pd.Series([1,2,3]), a = pd.Series([1,2,3])) == pd.Series
    assert pd2np(exc = None)(lambda x, a: type(a))(x = pd.Series([1,2,3]), a = pd.Series([1,2,3])) == np.ndarray
    assert pd2np(lambda x, a: type(a))(x = pd.Series([1,2,3]), a = pd.Series([1,2,3])) == np.ndarray
    

def test_tree_loop():
    weighted_sum = lambda d, w = {}: sum([d[key]*w.get(key,1) for key in d])
    assert weighted_sum(dict(x = 1, y = 2)) == 3
    assert weighted_sum(dict(x = 1, y = 2), w = dict(x = 0.4, y = 0.6)) == 1.6
    
    arg = dict(a = dict(x=1, y=1), b = dict(x = 1, y = 2, z = 3), c = dict(u = dict(x = 1, y = 2), v = dict(x = 1, y = 2, z = 3)))
    w = dict(a = dict(x = 0.1, y = 0.9), b = dict(x = 1, y = 1, z = 2), c = {})

    assert tree_loop(weighted_sum)(arg) == dict(a = 2, b = 6, c = dict(u = 3, v = 6))
    assert tree_loop(weighted_sum)(arg, w) == dict(a = 1, b = 9, c = dict(u = 3, v = 6))


def test_is_tree_deep():
    assert is_tree_deep(dict(a = 1), 0)
    assert is_tree_deep(dict(a = dict(b = 1)), 1)
    assert is_tree_deep(dict(a = dict(b = 1), c = dict(x = 1, y = 2)), 1)
    assert not is_tree_deep(dict(a = 1), 1)
    assert not is_tree_deep(dict(a = dict(b = 1), c = dict(x = 1, y = 2)), 2)
