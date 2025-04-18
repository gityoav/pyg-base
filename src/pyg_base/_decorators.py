import numpy as np
from pyg_base._logger import logger
from pyg_base._inspect import getargspec, getargs, argspec_add, getcallarg
from pyg_base._dictattr import dictattr
from pyg_base._as_list import as_list
from pyg_base._types import NoneType
from copy import copy
import datetime
import time
from inspect import FullArgSpec


__all__ = ['wrapper', 'try_back', 'try_nan', 'try_none', 'try_zero', 'try_false', 'try_true', 'try_list', 'timer']
_function = 'function'
_spec = 'function_fullargspec'


class DictArgSpec(dictattr):
    pass

def as_DictArgSpec(argspec):
    return DictArgSpec(args = argspec.args,
                        varargs=argspec.varargs, 
                        varkw=argspec.varkw, 
                        defaults=argspec.defaults, 
                        kwonlyargs=argspec.kwonlyargs, 
                        kwonlydefaults=argspec.kwonlydefaults, 
                        annotations=argspec.annotations)


class wrapper(dictattr):
    """
    A base class for all decorators. It is similar to functools.wraps but better. See below why wrapt cannot be used...
    You basically need to define the wrapped method and everything else is handled for you.
    - You can then use it either directly to decorate functions
    - Or use it to create parameterized decorators
    - the __name__, __wrapped__, __doc__ and the getargspec will all be taken care of.
    
    :Example:
    -------
    >>> class and_add(wrapper):
    >>>     def wrapped(self, *args, **kwargs):
    >>>         return self.function(*args, **kwargs) + self.add ## note that we are assuming self.add exists
    
    >>> @and_add(add = 3) ## create a decorator and decorate the function
    >>> def f(a,b):
    >>>     return a+b
        
    >>> assert f.add == 3
    >>> assert f(1,2) == 6
    

    Alternatively you can also use it this directly:
        
    >>> def f(a,b):
    >>>     return a+b
    >>> 
    >>> assert and_add(f, add = 3)(1,2) == 6

    
    :Example: Explicit parameter construction
    -----------------------------------------
    
    You can make the init more explict, also adding defaults for the parameters:

    >>> class and_add_version_2(wrapper):
    >>>     def __init__(self, function = None, add = 3):
    >>>         super(and_add, self).__init__(function = function, add = add)
    >>>     def wrapped(self, *args, **kwargs):
    >>>         return self.function(*args, **kwargs) + self.add

    >>> @and_add_version_2
    >>> def f(a,b):
    >>>     return a+b
    >>> assert f(1,2) == 6
        

    :Example: No recursion
    ------------------
    The decorator is designed to have a single instance of a specific wrapper
    
    >>> f = lambda a, b: a+b
    >>> assert and_add(and_add(f)) == and_add(f)

    This holds even for multiple levels of wrapping:

    >>> x = try_none(and_add(f))
    >>> y = try_none(and_add(x))
    >>> assert x == y        
    >>> assert x(1, 'no can add') is None        

    :Example: wrapper vs wrapt
    --------------------------
    wrapt (wrapt.readthedocs.io) is an awesome wrapping tool. If you have static library functions, none is better.
    The problem we face is that wrapt is too good in pretending the wrapped up object is the same as original function:
        
    >>> import wrapt    
    >>> def add_value(value):
    >>>     @wrapt.decorator
    >>>     def wrapper(wrapped, instance, args, kwargs):
    >>>         return wrapped(*args, **kwargs) + value
    >>>     return wrapper

    >>> def f(x,y):
    >>>     return x*y

    >>> add_three = add_value(value = 3)(f)
    >>> add_four = add_value(value = 4)(f)
    >>> assert add_four(3,4) == 16 and add_three(3,4) == 15

    >>> ## but here is the problem:
    >>> assert encode(add_three) == encode(add_four) == encode(f)
    
    So if we ever encode the function and send it across json/Mongo, the wrapping is lost and the user when she receives it cannot use it

    >>> class add_value(wrapper):
    >>>     def wrapped(self, *args, **kwargs):
    >>>         return self.function(*args, **kwargs) + self.value

    >>> add_three = add_value(value = 3)(f)
    >>> add_four = add_value(value = 4)(f)
    >>> encode(add_three)
    >>> {'value': 3, 'function': '{"py/function": "__main__.f"}', '_obj': '{"py/type": "__main__.add_value"}'}
    >>> encode(add_three)
    >>> {'value': 4, 'function': '{"py/function": "__main__.f"}', '_obj': '{"py/type": "__main__.add_value"}'}
 
    """    
    def __init__(self, function = None, *args, **kwargs):
        function = copy(function)
        if type(function) == type(self):
            kw = function._kwargs
            kw.update(kwargs)
            function = function.function
        else:
            kw = kwargs
        f = function
        while isinstance(f, wrapper):
            if type(f.function) == type(self):
                kw = f.function._kwargs
                kw.update(kwargs)
                f[_function] = f.function.function
            else:
                f = f.function

        super(wrapper, self).__init__(*args, **kw)
        self[_spec] = None
        self[_function] = function
        bad_keys = [key for key in kwargs if key.startswith('_')]
        if len(bad_keys):
            raise ValueError('Cannot wrap _hidden parameters %s'%bad_keys)
        for attr in ['doc']:
            attr = '__%s__'%attr
            if hasattr(self.function, attr):
                setattr(self, attr, getattr(self.function, attr))
                        

    @property
    def __name__(self):
        return getattr(self[_function], '__name__', 'pyg_base.wrapper(unnamed)')
    
    @property
    def __wrapped__(self):
        return self[_function]

    @property
    def fullargspec(self):
        if self[_spec] is None:
            self[_spec] = as_DictArgSpec(getargspec(self[_function]))
        return self[_spec]

    def __repr__(self):
        return '%s(%s)'%(self.__class__.__name__, dict(self - 'wrapper_function_spec'))

    __str__ = __repr__ 

    @property
    def _kwargs(self):
        return {key: value for key, value in self.items() if key!=_function and key!=_spec}

    def __call__(self, *args, **kwargs):
        if self[_function] is None and len(args) == 1 and len(kwargs) == 0:
            return type(self)(function = args[0], **self._kwargs)
        else:
            return getattr(self, 'wrapped', self[_function])(*args, **kwargs)
    
class try_back(wrapper):
    """
    wraps a function to try an evaluation. If an exception is thrown, returns first argument

    :Example:
    --------------
    >>> f = lambda a: a[0]
    >>> assert try_back(f)('hello') == 'h' and try_back(f)(5) == 5
    """
    def __init__(self, function = None, function_fullargspec = None):
        super(try_back, self).__init__(function = function, function_fullargspec = function_fullargspec)
    def wrapped(self, *args, **kwargs):
        try:
            return self.function(*args, **kwargs)
        except Exception:
            return args[0] if len(args)>0 else kwargs[getargs(self.function)[0]]
            

class try_value(wrapper):
    """
    wraps a function to try an evaluation. If an exception is thrown, returns a cached argument
    
    :Parameters:
    ------------
    function callable
        The function we want to decorate
    value: 
        If the function fails, it will return value instead. Default is None
    verbose: bool
        If set to True, the logger will warn with the error message.

    There are various convenience functions with specific values
    try_zero, try_false, try_true, try_nan and try_none will all return specific values if function fails.
    
    :Example:
    --------------
    >>> from pyg import *
    >>> f = lambda a: a[0]
    >>> assert try_none(f)(4) is None
    >>> assert try_none(f, 'failed')(4) == 'failed'

    
    """
    def __init__(self, function = None, repeat = 0, sleep = 0, return_value = True, value = None, verbose = None, function_fullargspec = None):
        super(try_value, self).__init__(function = function, repeat = repeat, sleep = sleep, return_value = return_value, value = value, verbose = verbose, function_fullargspec = None)
    def wrapped(self, *args, **kwargs):
        for i in range(self.repeat):
            try: 
                return self.function(*args, **kwargs)
            except Exception:
                if self.sleep:
                    time.sleep(self.sleep)
        if self.return_value:
            try: 
                return self.function(*args, **kwargs)
            except Exception as e:
                if self.verbose:
                    logger.warning('WARN: %s' % e)
                return copy(self.value)
        else:
            return self.function(*args, **kwargs)

try_nan = try_value(value = np.nan)
try_zero = try_value(value = 0)
try_none = try_value
try_true = try_value(value = True)
try_false = try_value(value = False)
try_list = try_value(value = [])


class do_if(wrapper):
    """
    Conditional execution based on the type of the first parameter
    
    :Parameters:
    ------------
    function callable
        The function we want to decorate
    inc: type or tuple of types
        If provided, executes the function only if first arg IS of the included types
    exc: type or tuple of types
        If provided, execute function only if first arg is NOT on of the excluded types.

    if_not_none will run if first parameter is not None
    
    :Example:
    --------------
    >>> from pyg import *
    >>> first_element = lambda a: a[0]
    >>> assert if_not_none(first_element)(None) is None
    >>> assert do_if(first_element, (str, list))('works on strings') == 'w'
    >>> assert do_if(first_element, (str, list))(['works','on','list']) == 'works'
    >>> assert do_if(first_element, (str, list))(('not', 'working', 'on', 'tuple')) == ('not', 'working', 'on', 'tuple')
    
    """
    def __init__(self, function = None, inc = None, exc = None, function_fullargspec = None):
        super(do_if, self).__init__(function = function, inc = inc, exc = exc, function_fullargspec = None)
    def wrapped(self, *args, **kwargs):
        if self.inc is None and self.exc is None: 
            return self.function(*args, **kwargs)
        arg = getcallarg(self.function, args, kwargs)
        if self.inc is not None and not isinstance(arg, self.inc):
            return arg
        elif self.exc is not None and isinstance(arg, self.exc):
            return arg
        else:
            return self.function(*args, **kwargs)

if_not_none = do_if(exc = NoneType)

def _str(value):
    """
    returns a short string:
    >>> _str([1,2,3])
    :Parameters:
    ----------------
    value : TYPE
        DESCRIPTION.

    :Returns:
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(value, (int, str, float, bool, datetime.datetime, datetime.date, type)):
        return str(value) 
    elif hasattr(value, '__len__'):
        return '%s[%i]'%(type(value), len(value))
    else:
        return str(type(value))
        
_txt = 'TIMER:%r args:[%r, %r] (%i runs) took %s sec'    
class timer(wrapper):
    """
    timer is similar to timeit but rather than execution of a Python statement, 
    timer wraps a function to make it log its evaluation time before returning output
    
    :Parameters:
    ------------
    function: callable
        The function to be wraooed 
    
    n: int, optional
        Number of times the function is to be evaluated. Default is 1
    
    time: bool, optional
        If set to True, function will return the TIME it took to evaluate rather than the original function output.


    :Example:
    ---------
    >>> from pyg import *; import datetime
    >>> f = lambda a, b: a+b
    >>> evaluate_100 = timer(f, n = 100, time = True)(1,2)
    >>> evaluate_10000 = timer(f, n = 10000, time = True)(1,2)
    >>> assert evaluate_10000> evaluate_100
    >>> assert isinstance(evaluation_time, datetime.timedelta)
    """
    
    def __init__(self, function, n = 1, time = False, function_fullargspec = None):
        super(timer, self).__init__(function = function, n = n, time = time, function_fullargspec = function_fullargspec)

    def wrapped(self, *args, **kwargs):
        t0 = datetime.datetime.now()
        for _ in range(self.n):
            res = self.function(*args, **kwargs)
        t1 = datetime.datetime.now()
        time = t1 - t0
        logger.info(_txt%(getattr(self.function,'__name__', ''), 
                            [_str(a) for a in args],
                            ['%s=%s'%(key, _str(value)) for key, value in kwargs.items()],
                            self.n,
                            time
                            ))        
        return time if self.time else res

class kwargs_support(wrapper):
    """
    Extends a function to support **kwargs inputs
    
    :Example:
    ---------
    >>> from pyg import *
    >>> @kwargs_support
    >>> def f(a,b):
    >>>     return a+b
    
    >>> assert f(1,2, what_is_this = 3, not_used = 4, ignore_this_too = 5) == 3
    
    """
    def __init__(self, function = None, function_fullargspec = None):
        super(kwargs_support, self).__init__(function = function, function_fullargspec = function_fullargspec)
    
    @property
    def _args(self):
        return getargs(self.function)
        
    def wrapped(self, *args, **kwargs):
        _args = self._args
        kwargs = {key : value for key, value in kwargs.items() if key in _args}
        return self.function(*args, **kwargs)
 

class kwpartial(wrapper):
    """
    One of the problems of functions with kwargs support is that dictable does not 
    know what parameters to present to it.
    
    >>> from pyg import *
    >>> f = lambda **kwargs: len(kwargs)
    >>> assert Dict(a = 1, b = 2)[f] == 0 ## no parameters are presented to f

    kwpartial acts as a partial function, but it also able to specify what kw are presented to it...
        
    >>> f = lambda a, b, c=1, **kwargs: len(kwargs)
    >>> self = kwpartial(f, kw = ['x', 'y'], b = 1, c = 2, z = 4)
    >>> assert self(a = 1) == 1 # just z is in kwargs
    >>> d = Dict(a = 1, x = 2)
    >>> assert d[self] == 2 # z and x
    >>> d = Dict(a = 1, x = 2, w = 0)
    >>> assert d[self] == 2 # z and x
    >>> d = Dict(a = 1, x = 2, w = 0, y=3)
    >>> assert d[self] == 3 # z and x and y are presented

    """
    def __init__(self, function = None, kw = None, function_fullargspec = None, **keywords):
        argspec = getargspec(function)
        if argspec.varkw is None:
            raise ValueError('kwpartial only supports function with **kwargs support')
        if argspec.varargs is not None:
            raise ValueError('kwpartial only support *args-less function, but function has *%s'%argspec.varargs)
        super(kwpartial, self).__init__(function = function, keywords = keywords, kw = kw, function_fullargspec = function_fullargspec)

    @property
    def fullargspec(self):
        spec = getargspec(self.function)
        args = [a for a in as_list(self.kw) if a not in spec.args] 
        if len(args) == 0:
            return spec
        else:
            return FullArgSpec(args=args + spec.args, 
                               varargs=spec.varargs, varkw=spec.varkw, defaults=spec.defaults, 
                               kwonlyargs=spec.kwonlyargs, 
                               kwonlydefaults=spec.kwonlydefaults, 
                               annotations=spec.annotations)
    
    def wrapped(self, *args, **kwargs):
        kwargs.update(self.keywords)
        return self.function(*args, **kwargs)


 
    