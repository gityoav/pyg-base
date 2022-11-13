import pandas as pd; import numpy as np
from pyg_base._inspect import getargs, getcallarg, getcallargs, getargspec, call_with_callargs
from pyg_base._types import is_ts, is_str, is_df, is_pd, is_series, is_arr, is_array, is_tuple, is_dict
from pyg_base._decorators import wrapper
from pyg_base._as_list import as_tuple, as_list

__all__ = ['loop', 'loops', 'len0', 'pd2np', 'shape', 'loop_all']


def _zero():
    return 0

def len0(value):
    """
    returns the len of an object or 0 if no len(value) exists
    
    :Example:
    ---------
    >>> assert len0(5) == 0
    >>> assert len0('a string is dimensionless') == 0
    >>> assert len0([]) == 0
    >>> assert len0([1,2]) == 2

    Parameters
    ----------
    value : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    try:
        return 0 if is_str(value) else getattr(value, '__len__', _zero)()
    except Exception:
        return 0


def shape(value):
    return getattr(value, 'shape', ())
 

def _item_by_key(value, key, keys, i = None):
    if isinstance(value, dict):
        if sorted(value.keys()) == keys:
            return value[key]
        else:
            return type(value)({k : _item_by_key(v, key, keys, i) for k, v in value.items()})
    elif isinstance(value, pd.Series) and sorted(value.index.values) == keys:
        return value[key] 
    elif isinstance(value, pd.DataFrame) and sorted(value.columns) == sorted(keys):
        return value[key]
    elif isinstance(value, pd.DataFrame) and sorted(value.index) == sorted(keys):
        return value.loc[key]
    elif is_array(value) and len(value.shape):
        if len(value.shape) == 2 and value.shape[1] == len(keys) and i is not None:
            return value.T[i]
        elif len(value.shape) == 1 and value.shape[0] == len(keys) and i is not None:
            return value[i]
        else:
            return value
    else:
        if i is not None and len0(value) == len(keys):
            return value[i]
        else:
            return value

def _item_by_i(value, i, n):
    if is_df(value) and value.shape[1] == 1:
        value = value.iloc[:,0]
    elif is_array(value) and len(value.shape) == 2 and value.shape[1] == 1:
        value = value.T[0]
    if isinstance(value, (list, tuple)):
        if len(value) == n:
            return value[i]
        else:
            return type(value)([_item_by_i(v, i, n) for v in value])
    elif is_array(value):
        if len(value.shape) == 2 and value.shape[-1] == n:
            return value.T[i]
        if len(value.shape) == 1 and value.shape[-1] == n:
            return value[i]
    elif is_series(value) and len(value) == n and not is_ts(value):
        return value.iloc[i]
    elif is_df(value) and value.shape[1] == n:
        return value.iloc[:, i]
    return value           

# add_index_and_columns = _np2pd

def add_index_and_columns(res, arg):
    if isinstance(res, tuple):
        return tuple([add_index_and_columns(r, arg) for r in res])
    elif isinstance(res, dict):
        return type(res)({k: add_index_and_columns(r, arg) for k,r in res.items()})
    if isinstance(res, (pd.Series, pd.DataFrame)) and len(res) == len(arg):
        res.index = arg.index
    if isinstance(res, (pd.DataFrame)) and res.shape[1] == arg.shape[1]:
        res.columns = arg.columns
    return res

def axis0_to_dataframe(res, arg):
    if min([is_tuple(r) for r in res], default = False) and len(set([len(r) for r in res])) == 1:
        return tuple([axis0_to_dataframe(r, arg) for r in list(zip(*res))])
    elif min([is_dict(r) for r in res], default = False) and len(set([len(r) for r in res])) == 1:
        keys = res[0].keys()
        tp = type(res[0])
        return tp({key : axis0_to_dataframe([r[key] for r in res], arg) for key in keys})
    try:
        res = type(arg)(dict(zip(arg.columns, res)))
        return add_index_and_columns(res, arg)
    except ValueError:
        res = pd.Series(res)
        res.index = arg.columns
        return res

def axis0_to_array(res, arg):
    if min([is_tuple(r) for r in res], default = False) and len(set([len(r) for r in res])) == 1:
        return tuple([axis0_to_array(r, arg) for r in list(zip(*res))])
    elif min([is_dict(r) for r in res], default = False) and len(set([len(r) for r in res])) == 1:
        keys = res[0].keys()
        tp = type(res[0])
        return tp({key : axis0_to_array([r[key] for r in res], arg) for key in keys})
    rtn = np.array(res)
    if len(rtn.shape)>1:
        rtn = rtn.T
    return rtn

class loops(wrapper):

    """
    
    converts a function to loop over the arguments, depending on the type of the first argument

    :Examples:
    --------------
    >>> @loop(dict, list, pd.DataFrame, pd.Series)
    >>> def f(a,b):
    >>>     return a+b
    
    >>> assert f(1,2) == 3
    >>> assert f([1,2,3],2) == [3,4,5]
    >>> assert f([1,2,3], [4,5,6]) == [5,7,9]    
    
    >>> assert f(dict(x=1,y=2), 3) == dict(x = 4, y = 5)
    >>> assert f(dict(x=1,y=2), dict(x = 3, y = 4)) == dict(x = 4, y = 6)
    

    >>> a = pd.Series(dict(x=1,y=2))
    >>> b = dict(x=3,y=4)
    >>> assert np.all(f(a,b) == pd.Series(dict(x=4,y=6)))    

    >>> a = pd.DataFrame(dict(x=[1,1],y=[2,2])); a.index = [5,10]
    >>> b = dict(x=3,y=4)
    >>> res =  f(a,b)
    >>> assert np.all(res == pd.DataFrame(dict(x=[4,4],y=[6,6]), index = [5,10]))    
    
    >>> a = pd.DataFrame(dict(x=[1,1],y=[2,2])); a.index = [5,10]
    >>> res =  f(a,[3,4])
    >>> assert np.all( res == pd.DataFrame(dict(x=[4,4],y=[6,6]), index = [5,10]))   
    
    >>> from pyg import * 
    >>> @loop_all
    >>> def f(x, y, col):
    >>>     return x * y[col]
    >>> x = dict(a = 1, b = 2)
    >>> y = pd.DataFrame(dict(col1 = [1,2], col2 = [3,4]), index = ['a', 'b'])
    >>> assert f(x, y, 'col1') == dict(a = 1, b = 4)
    >>> assert f(x, y, 'col2') == dict(a = 3, b = 8)
    
    """

    def __init__(self, function = None, types = None):
        super(loops, self).__init__(function = function, types = as_tuple(types))

    @property
    def first(self):
        args = getargs(self)
        return args[0] if len(args) else None
    
    def T(self, arg, args, kwargs):
        arg_ = _T(arg)
        args_ = _T(args)
        kwargs_ = _T(kwargs)
        res = self._wrapped(arg_, args_, kwargs_)
        return _T(res)

    def wrapped(self, *args, **kwargs):
        top = self.first
        if len(args) == 0 and not top in kwargs:
            return self.function(*args, **kwargs)
        if len(args):
            arg, args_, kwargs_ = args[0], args[1:], kwargs
        else:
            arg = kwargs.pop(top)
            args_, kwargs_ = args, kwargs
        if isinstance(arg, pd.Series) and pd.Series in self.types and not is_ts(arg):
            keys = sorted(arg.index)
            res = {key : self._wrapped(arg[key], (_item_by_key(a,key,keys) for a in args_), {k : _item_by_key(v,key,keys) for k,v in kwargs_.items()}) for key in arg.index}
            return type(arg)(res)          
        else:
            return self._wrapped(arg, args_, kwargs_)
    
    def _wrapped(self, arg, args, kwargs):
        axis = kwargs.pop('axis', 0)
        if isinstance(arg, dict) and dict in self.types:
            keys = sorted(arg.keys())
            res = {key : self._wrapped(arg[key], (_item_by_key(a,key,keys) for a in args), {k : _item_by_key(v,key,keys) for k,v in kwargs.items()}) for key in arg.keys()}
            return type(arg)(res)
        elif isinstance(arg, pd.DataFrame) and pd.DataFrame in self.types:
            if axis in (1,-1):
                res = self.T(arg, args, kwargs)
                if isinstance(res, (pd.DataFrame, pd.Series)) and len(res) == len(arg):
                    res.index = arg.index
                return add_index_and_columns(res, arg)
            else:
                keys = sorted(arg.columns)
                res = [self._wrapped(arg[key], (_item_by_key(a,key,keys,i) for a in args), {k : _item_by_key(v,key,keys,i) for k,v in kwargs.items()}) for i, key in enumerate(arg.columns)]
                rtn = axis0_to_dataframe(res, arg)
                return rtn
        elif isinstance(arg, pd.Series):
            return self.function(arg, *args, **kwargs)
        elif isinstance(arg, np.ndarray) and np.ndarray in self.types:
            if len(arg.shape) <= 1:
                return self.function(arg, *args, **kwargs) 
            else:
                if axis in (1,-1):
                    return self.T(arg, args, kwargs)
                else:
                    n = arg.shape[1]
                    res = [self._wrapped(_item_by_i(arg,i,n), (_item_by_i(a,i,n) for a in args), {k: _item_by_i(v,i,n) for k, v in kwargs.items()}) for i in range(n)]
                    return axis0_to_array(res, arg)
        elif isinstance(arg, self.types):
            n = len(arg)
            res = [self._wrapped(arg[i], (_item_by_i(a,i,n) for a in args), {k: _item_by_i(v,i,n) for k, v in kwargs.items()}) for i in range(n)]                            
            return type(arg)(res)
        else:
            return self.function(arg, *args, **kwargs)

def loop(*types):
    """
    returns an instance of loops(types = types)
    """
    return loops(types = as_tuple(types))


def _T(arg):
    if isinstance(arg, tuple):
        return type(arg)([_T(a) for a in arg])
    elif isinstance(arg, list):
        return type(arg)([_T(a) for a in arg])
    elif isinstance(arg, dict):
        return type(arg)({k :_T(a) for k,a in arg.items()})
    else:
        return arg.T if isinstance(arg, (np.ndarray, pd.DataFrame)) and len(arg.shape) == 2 else arg

def _values(a):
    if isinstance(a, (list, tuple)):
        return type(a)([_values(v) for v in a])
    elif isinstance(a, dict):
        return type(a)({k : _values(v) for k,v in a.items()})
    if is_series(a):
        return a.values
    elif is_df(a):
        if a.shape[-1] == 1:
            return a.values.T[0]
        else:
            return a.values
    else:
        return a

def _np2pd(res, arg):
    if isinstance(res, tuple):
        return tuple([_np2pd(r, arg) for r in res])
    elif isinstance(res, tuple):
        return [_np2pd(r, arg) for r in res]
    elif isinstance(res, dict):
        return type(res)({k: _np2pd(r, arg) for k,r in res.items()})
    n = len0(res)
    if n == len0(arg):
        if len(res.shape) == 2:
            res = pd.DataFrame(res, arg.index)
            if len(arg.shape) == 2 and res.shape[1] == arg.shape[1]:
                res.columns = arg.columns
            return res
        else:
            return pd.Series(res, arg.index)
    elif is_df(arg) and n == arg.shape[1]:
        return pd.Series(arg.columns, res)
    else:
        return res

class pd2np(wrapper):
    """
    converts a numpy-based function to work on pandas by converting to a numpy arrage and then converting back to panadas dataframe
    """
    def wrapped(self, *args, **kwargs):
        arg = getcallarg(self.function, args, kwargs)
        if not is_pd(arg):
            return self.function(*args, **kwargs)
        args_ = [_values(a) for a in args]
        kwargs_ = {k : _values(v) for k, v in kwargs.items()}
        res = self.function(*args_, **kwargs_)
        return _np2pd(res, arg)


loop_all = loops(types = (pd.Series, pd.DataFrame, np.ndarray, list, tuple, dict))


@loop(dict, list, tuple)
def _cut_pd(arg, index):
    if is_pd(arg):
        return arg.drop(index = index)
    elif isinstance(arg, pd.Index):
        return arg.drop(index) # index just has labels
    else:
        return arg

@loop(dict, list, tuple)
def _cut_np(arg, n, t = None):
    if is_arr(arg) and (t is None or len(arg) == t):
        return arg[n:]
    else:
        return arg



class skip_if_data_pd(wrapper):
    """
    If a function has a 'data' parameter which is provided and is a timeseries, use it to start running from that point onwards
    
    :Example:
    ---------
    >>> a = pd.Series(range(6))
    >>> f = skip_if_data_pd(lambda a: a ** 2)    
    >>> assert list(f(a).values) == [0,1,4,9,16,25]
    
    >>> res = f(a, data = a.iloc[:4]) ## data is provided and is assumed to be the history of past calculations, regardless if it is actual value
    >>> assert list(res.values) == [0,1,2,3] + [16,25]
    
    """
    def wrapped(self, *args, **kwargs):
        data = kwargs.pop('data', None)
        if is_pd(data):
            data = data.iloc[:-1]
            index = data.index
            args, kwargs = _cut_pd((args, kwargs), index = index)
        res = self.function(*args, **kwargs)
        if res is None: ## error handling
            return None
        if is_pd(data):
            if not is_pd(res):
                raise ValueError('if data parameter %s is a dataframe, it is part of the answer so result must also be a dataframe %s'%(data, res))                    
            return pd.concat([data,res]).sort_index()
        else:
            return res

def _np2pd_(res, index = None, columns = None):
    if is_pd(res) or not is_arr(res) or len(res.shape)>2 or (columns is None and index is None):
        return res
    elif len(res.shape) == 1:
        if index is not None and len(res) == len(index):
            return pd.Series(res, index)
        if columns is not None and len(res) == len(columns):
            return pd.Series(res, columns)
        else:
            return res
    elif len(res.shape) == 2:
        return pd.DataFrame(data = res, columns = columns, index = index) 
    else:
        return res

class skip_if_data_pd_or_np(wrapper):
    """
    If a function has a 'data' parameter which is provided and is a timeseries or a numpy array, use it to start running from that point onwards
    
    :Example:
    ---------
    >>> a = np.array([0,1,2,3,4,5])
    >>> f = skip_if_data_pd_or_np(lambda a: a ** 2)    
    >>> assert list(f(a)) == [0,1,4,9,16,25]
    
    >>> res = f(a, data = a[:4]) ## data is provided and is assumed to be the history of past calculations, regardless if it is actual value
    >>> assert list(res) == [0,1,2,3] + [16,25]
    
    """
    def wrapped(self, *args, **kwargs):
        data = kwargs.pop('data', None)
        arg = getcallarg(self.function, args, kwargs)
        index = kwargs.pop('index', arg.index if is_pd(arg) else None)
        columns = kwargs.pop('columns', arg.columns if is_df(arg) else None)
        if index is not None:
            index = pd.Index(index)
        if data is not None:
            t = None
            if index is not None:
                t = len(index)
            else:
                if is_arr(arg) or is_pd(arg):
                    t = len(arg)
            if is_pd(data): ## previous result is a timeseries
                args, kwargs, index = _cut_pd((args, kwargs, index), index = data.index)
            args, kwargs, index = _cut_np((args, kwargs, index), n = len(data), t = t)
        res = self.function(*args, **kwargs)        
        if res is None: ## error handling
            return None
        if is_arr(data):
            if not is_arr(res):
                raise ValueError('if data parameter %s is a numpy array, it is part of the answer so result must also be a dataframe %s'%(data, res))                                    
            return np.concatenate([data,res])            
        res_df = _np2pd_(res, index, columns)
        if is_pd(data):
            if not is_pd(res_df):
                raise ValueError('if data parameter %s is a dataframe, it is part of the answer so result must also be a dataframe %s'%(data, res_df))                                    
            return pd.concat([data,res]).sort_index()
        return res_df

def _getitem(value, param, strict = False):
    if isinstance(value, dict) and param in value.keys():
        return value[param]
    elif isinstance(value, pd.Series) and param in value.index:
        return value[param]
    elif isinstance(value, pd.DataFrame) and param in value.columns:
        return value[param]
    elif isinstance(value, pd.DataFrame) and param in value.index:
        return value.loc[param]
    else:
        if strict and isinstance(value, (pd.Series, pd.DataFrame, dict)):
            raise ValueError('could not find parameter %s in %s'%(param, value))
        else:
            return value

class grab_parameter_from_dict(wrapper):
    """
    allows the function to provide a "data factory" for a function.
    
    
    Example
    -------
    >>> @grab_parameter_from_dict
    >>> def f(a,b):
    >>>     return a+b
    
    >>> data_factory = dict(a = 1, b = 2)
    >>> assert f(a = data_factory, b = data_factory) == 3
    
    >>> data_factory = pd.Series(dict(a = 10, b = 20))
    >>> assert f(a = data_factory, b = data_factory) == 30
    
    :Example: being specific about what parameters are converted
    ---------    
    
    >>> @grab_parameter_from_dict(parameters = 'a') # convert just 'a'
    >>> def f(a,b):
    >>>     return a+b

    >>> data_factory = pd.Series(dict(a = 10, b = 20))
    >>> assert eq(f(a = data_factory, b = data_factory), data_factory['a'] + data_factory)
    
    ### From example, we see that there may be ambiguity on using the factory and conversion...
    
    :Example: being strict about which parameters are converted
    --------
    >>> @grab_parameter_from_dict(parameters = 'a', strict = False) # convert just 'a'
    >>> def f(a,b):
    >>>     return a+b
    
    >>> bad_data_factory = pd.Series(dict(c = 20))
    >>> f(a = bad_data_factory, b = 5) = bad_data_factory + 5
    >>> ## This is a problem, as 'a' should have converted to a value
    
    >>> @grab_parameter_from_dict(parameters = 'a', strict = True) # convert just 'a', be strict
    >>> def f(a,b):
    >>>     return a+b
    
    >>> import pytest
    >>> with pytest.raises(ValueError):
    >>>     f(a = bad_data_factory, b = 5)

    >>> assert f(a = 3, b = 5) == 8
    >>> #This is still OK as 'a' is not a type dict/pd.Series or pd.DataFrame

    :Example: remapping the parameters
    ---------
    >>> @grab_parameter_from_dict(a = 'fancy_name') # convert just 'a' using 'fancy_name' in data factory
    >>> def f(a,b):
    >>>     return a+b
    >>>
    >>> factory = dict(fancy_name = 4)
    >>> assert f(a = factory, b = 3) == 7
    
    """
    
    def __init__(self, function = None, parameters = None, strict = None, **kwargs):
        if strict is None:
            strict = False if parameters is None else True
        if isinstance(parameters, (list, str)):
            parameters = as_list(parameters)
            parameters = dict(zip(parameters, parameters))
        if kwargs:
            parameters = parameters or {}
            parameters.update(kwargs)
        super(grab_parameter_from_dict, self).__init__(function = function, parameters = parameters, strict = strict)

    def wrapped(self, *args, **kwargs):
        spec = getargspec(self.function)
        callargs = getcallargs(self.function, *args, **kwargs)
        strict = self.strict
        if self.parameters is None:
            callargs = {param: value if param == spec.varargs else _getitem(value, param, strict = strict) for param, value in callargs.items()}
            result = call_with_callargs(self.function, callargs)
        else:
            parameters = self.parameters
            callargs = {param: _getitem(value, parameters[param], strict = strict) if param in parameters else value for param, value in callargs.items()}
            result = call_with_callargs(self.function, callargs)
        return result
