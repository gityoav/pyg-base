from collections.abc import Hashable
from pyg_base._types import is_primitive
from pyg_base._decorators import wrapper, getargs

_cache = 'cache'
def _prehash(value):
    if isinstance(value, (tuple,list)):
        return tuple([_prehash(v) for v in value])
    elif isinstance(value, dict):
        try:
            return tuple(sorted([(k, _prehash(v)) for k, v in value.items()]))
        except TypeError:
            return tuple([(k, _prehash(v)) for k, v in value.items()])
    else:
        return value



def _hash(value):
    value = _prehash(value)
    if is_primitive(value):
        return value
    else:
        return hash(value)
    
class cache_func(wrapper):
    """
    >>> @cache
    >>> def f(a,b):
    >>>     return a+b

    >>> assert f(1,2) == 3
    >>> key = ((1, 2), ())
    >>> assert key in f.cache
    >>> assert f.cache[key] == 3

    """
    def _key(self, *args, **kwargs):
        return _prehash((args, kwargs))

    def wrapped(self, *args, **kwargs):
        key = self._key(*args, **kwargs)
        try:
            self.cache = getattr(self, _cache, {})
            if key not in self.cache:
                self.cache[key] = self.function(*args, **kwargs)
            return self.cache[key]
        except Exception:
            return self.function(*args, **kwargs)
        
    def clear_cache(self):
        self.cache = {}
        return self
    


def cache(function):
    """
    :Example:
    ---------
    >>> @cache
    >>> def f(a,b):
    >>>     return a+b

    >>> assert f(1,2) == 3
    >>> key = ((1, 2), ())
    >>> assert key in f.cache
    >>> assert f.cache[key] == 3

    """
    args = getargs(function)
    if args and args[0] in ('self', 'cls'):
        raise ValueError('cannot cache method')
    else:
        return cache_func(function)
    


    
