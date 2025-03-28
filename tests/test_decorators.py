from pyg_base import getargs, getargspec, dictable, dt, try_none, try_back, try_zero, presync, wrapper, eq, try_list, try_true, try_false, try_nan, try_value, kwargs_support
from pyg_base._decorators import _str , _spec

import numpy as np
import pytest


def test_cache_fullargspec():
    f = try_none(lambda v: v)
    assert f[_spec] is None 
    args = getargs(f)
    assert args == ['v']
    assert f[_spec] == getargspec(lambda v: v)
    assert f.fullargspec == getargspec(lambda v: v)
    assert getargs(f) == ['v']
    f = try_none()
    assert getargs(f) == []
    assert f[_spec] is None
    f = f(lambda v: v)    
    assert f[_spec] is None
    assert f.fullargspec == getargspec(lambda v: v)
    assert f[_spec]== getargspec(lambda v: v)
    
def test_cache_fullargspec_in_chain():
    f = try_none(lambda v: v)
    assert getargs(f) == ['v']
    g = kwargs_support(f)
    assert g.function == f
    assert getargs(g) == getargs(f)
    assert g[_spec] == getargspec(lambda v: v)

    rs  = dictable(v = [1,2,3])
    rs = rs(w = g)
    assert rs.w == rs.v


def test_wrapper():
    class and_add(wrapper):
        def wrapped(self, *args, **kwargs):
            return self.function(*args, **kwargs) + self.add
    
    @and_add(add = 3)
    def f(a,b):
        return a+b
    
    assert isinstance(f, dict)
    assert f['add'] == f.add == 3
    assert f(1,2) == 6

    
    class and_add_version_2(wrapper):
        def __init__(self, function = None, add = 3):
            super(and_add_version_2, self).__init__(function = function, add = add)
        def wrapped(self, *args, **kwargs):
            return self.function(*args, **kwargs) + self.add

    @and_add_version_2
    def f(a,b):
        return a+b

    assert f(1,2) == 6
        

    f = lambda a, b: a+b
    assert and_add_version_2(and_add_version_2(f)) == and_add_version_2(f)

    x = try_none(and_add_version_2(f))
    y = try_none(and_add_version_2(x))
    assert x == y        
    assert x(1, 'no can add') is None        


def test_try():
    f = lambda a: a[0]
    for t, v in [(try_none, None), (try_zero,0), (try_nan,np.nan), (try_true, True), (try_false, False), (try_list, [])]:
        assert eq(t(f)(5), v)
    assert try_value(f, verbose = True)(4) is None
    assert try_value(f, return_value = 'should log', verbose = True)(4) == 'should log'
    
    
        
def test_try_back():
    def f(a):
        return a[0]
    assert try_back(f)(5) == 5
    assert try_back(f)('hello') == 'h'
    assert try_back(f).__wrapped__ == f
    assert try_back(f).__repr__().startswith("try_back({'function':")
    
def test_fail_with_hidden_params():
    def f(_a):
        return _a
    class funny(wrapper):
        pass
    assert funny(exposed = 1)(f)(4)==4
    with pytest.raises(ValueError):
        funny(_hidden= 1)(f)


def test_decorators__str():
    for value in [1, '2', 3.0, True, dt(0)]:
        assert _str(value) == str(value)
    assert _str(None) == "<class 'NoneType'>"
        
