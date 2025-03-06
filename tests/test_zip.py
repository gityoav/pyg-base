from pyg_base import zipper, lens
import numpy as np
import pytest

def test_zipper():
    assert list(zipper([1,2,3], 4)) == [(1, 4), (2, 4), (3, 4)] 
    assert list(zipper([1,2,3], [4,5,6])) == [(1, 4), (2, 5), (3, 6)] 
    assert list(zipper([1,2,3], [4,5,6], [7])) ==  [(1, 4, 7), (2, 5, 7), (3, 6, 7)] 
    assert list(zipper([1,2,3], [4,5,6], None)) ==  [(1, 4, None), (2, 5, None), (3, 6, None)] 
    assert list(zipper((1,2,3), np.array([4,5,6]), None)) ==  [(1, 4, None), (2, 5, None), (3, 6, None)] 
    with pytest.raises(ValueError):
        zipper([1,2,3,4], [1,2,3])
    
    for d in [dict(a = 1), dict(a = 1, b = 2), dict(a = 1, b = 2, c = 3)]:
        assert dict(zipper(d.keys(), d.values())) == d

    assert list(zipper(np.array([1]), np.array([2]))) == [(1,2)]
    assert list(zipper(np.array([1]), np.array([1,2,3]))) == [(1, 1), (1, 2), (1, 3)]
    assert list(zipper(dict(a = 0).values(), np.array([1]), np.array([1,2,3]))) == [(0,1,1), (0,1,2), (0,1,3)]
    
    


def test_lens():
    assert lens([1,2,3,4], [1,2,3,4], [1,2,3,4]) == 4
    with pytest.raises(ValueError):
        lens([1,2,3,4], [1,2,3])
