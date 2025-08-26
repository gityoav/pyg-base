from _collections_abc import dict_keys, dict_values
from pyg_base._as_list import as_list, as_tuple
from pyg_base._as_float import as_float
from pyg_base._cache import cache
from pyg_base._dates import ndt
from pyg_base._decorators import kwargs_support, try_none, try_back
from pyg_base._dict import Dict
from pyg_base._file import read_csv
from pyg_base._inspect import getargs
from pyg_base._logger import logger
from pyg_base._sort import sort, cmp
from pyg_base._tree import is_tree, tree_to_table
from pyg_base._txt import lower
from pyg_base._types import is_str, is_strs, is_arr, is_df, is_dicts, is_int, is_ints, is_tuple, is_bools, is_nan, is_num
from pyg_base._zip import zipper, lens
from pyg_npy import pd_read_npy
from functools import reduce
from pathlib import Path

import pandas as pd
import re

Pattern = type(re.compile('')) ## this types definition moves across re. versions

__all__ = ['dict_concat', 'dictable', 'is_dictable']

@cache
def _print_cols(*lcols):
    logger.info('inner joining on %s. To stop message, specify columns explicitly in the join'%list(lcols))

    
def nan2none(v):
    return None if is_nan(v) or (is_str(v) and (len(v.strip()) == 0 or v[0] == '#' or v == 'n/a')) else v

def relabel_lower(v):
    if is_str(v):
        return v.strip().replace(' ', '_').replace('"','').replace("'",'').replace(',','').lower()
    else:
        return v    


def _dict_in_place_update(a, b):
    a.update(b)
    return a

def dict_concat(*dicts):
    """
    A method of turning a list of dicts into one happy dict sharing its keys

    :Parameters:
    ----------------
    *dicts : list of dicts
        a list of dicts

    :Returns:
    -------
    dict
        a dict of a list of values.
    
    :Example:
    --------------
    >>> dicts = [dict(a=1,b=2), dict(a=3,b=4,c=5)]
    >>> assert dict_concat(dicts) == dict(a = [1,3], b = [2,4], c = [None,5])
    >>> dicts = [dict(a=1,b=2)]
    >>> assert dict_concat(dicts) == dict(a = [1], b = [2])
    >>> assert dict_concat([]) == dict()
    """
    
    dicts = as_list(dicts)
    if len(dicts) == 0:
        return {}
    elif len(dicts) == 1:
        return {key: [value] for key, value in dicts[0].items()}  # a shortcut, we dont need to run a full merge
    possible_keys = list(set([tuple(sorted(d.keys())) for d in dicts]))
    if len(possible_keys) == 1:
        pairs = [sorted(d.items()) for d in dicts]
        keys = possible_keys[0]
        values = zip(*[[value for _, value in row] for row in pairs])
        res = dict(zip(keys, map(list, values)))
        return res
    else:
        keys = reduce(lambda res, keys: res | set(keys), possible_keys, set())
        return {key: [d.get(key) for d in dicts] for key in keys}
    

def _text_box(value, max_rows = 5, max_chars = 50):
    """
    a = dictable(a = range(10)) * dict(b = range(10))
    b = (dictable(a = range(5)) * dict(b = range(5))) * dict(c = range(10))
    self = c = dictable(key = ['a', 'b'] * 5, rs = [a,b]*5)
    """
    v = str(value)
    res = v.split('\n')
    if max_rows:
        res = res[:max_rows]
    if max_chars:
        res = [row[:max_chars] for row in res]
    return res


def _is_path(path):
    path = path.lower()
    return len([x for x in ['.xlsx', '.xls', '.pickle', '.dictable', '.parquet', '.csv', '.npy', '.npa'] if path.endswith(x)]) > 0
        

_modes = { 0:     lambda lhs,rhs:lhs,
          'lhs':  lambda lhs,rhs:lhs,
          'left': lambda lhs,rhs:lhs,
          'l':    lambda lhs,rhs:lhs,
          1:       lambda lhs,rhs:rhs,
          'rhs':   lambda lhs,rhs:rhs,
          'right': lambda lhs,rhs:rhs,
          'r' :    lambda lhs,rhs:rhs,
          'l|r': lambda lhs,rhs:rhs if lhs is None else lhs, ## left or right
          'r|l': lambda lhs,rhs:lhs if rhs is None else rhs, ## right or left
          }


def _data_columns_as_dict(data, columns = None):
    """
    >>> assert _data_columns_as_dict(data = [], columns = []) == dict()
    >>> assert _data_columns_as_dict([], 'a') == dict(a = [])
    >>> assert _data_columns_as_dict(data = [], columns = ['a','b']) == dict(a = [], b = [])
    
    :Parameters:
    ----------------
    data : TYPE
        DESCRIPTION.
    columns : TYPE, optional
        DESCRIPTION. The default is None.

    :Returns:
    -------
    TYPE
        DESCRIPTION.

    """
    if isinstance(data, zip):
        data = list(data)
    if data is None or (isinstance(data, list) and data == []):
        return {}
    elif isinstance(data, dict):
        return dict_concat(tree_to_table(data, columns)) if is_tree(columns) else dict(data)
    if is_str(data) and _is_path(data):
        data = Path(data)
    if isinstance(data, Path): ## convert Path by loading the data
        path = str(data).lower()
        if path.endswith('.csv'):
            data = read_csv(path)
        elif '.xls' in path:
            if is_str(columns):
                data = dict(dictable.read_excel(path, columns))
                columns = None
            else:
                data = dict(dictable.read_excel(path))
        elif path.endswith('.parquet'):
                data = pd.read_parquet(path)
        elif path.endswith('.pickle'):
            data = pd.read_pickle(path)
        elif path.endswith('.dictable'):
            try:
                data = pd.read_pickle(path)
            except Exception:
                data = pd.read_parquet(path)
        elif path.endswith('.npy'):
            data = pd_read_npy(path)
    if columns is not None:
        if is_str(columns):
            if (is_str(data) and '.xls' in data) or isinstance(data, pd.io.excel.ExcelFile):
                return dict(dictable.read_excel(data, columns))
            else:
                return {columns : data}
        else:
            return dict(zipper(columns, zipper(*data)))
    else:
        if is_str(data):
            return dict(data = data)
        if is_df(data):
            if data.index.name is not None:
                data = data.reset_index()
            return data.to_dict('list')
        tp = str(type(data)).lower()        
        if hasattr(data, 'next') and 'cursor' in tp:
            data = list(data)
        elif hasattr(data, 'find') and 'collection' in tp:
            data = list(data.find({}))
        if isinstance(data, list):
            if len(data) == 0:
                return {}
            elif min([is_tuple(row) and len(row) == 2 for row in data]):
                return dict(data)
            elif is_dicts(data):
                return dict_concat(data)
            elif min([isinstance(i, list) for i in data]):
                return dict(zipper(data[0], zipper(*data[1:])))
            else:
                return dict(data = data)
        elif isinstance(data, dict):
            return dict(data)
        else:
            return dict(data = data)


def _value(value):
    if value is None:
        return [None]
    elif isinstance(value, (dict_values, dict_keys, range)):
        return list(value)
    else:
        return list(value) if isinstance(value, tuple) else as_list(value)


def _row_check(row, key, value):
    v = row[key]
    if value is None:
        return v is None
    if is_nan(value):
        return is_nan(v)
    if isinstance(value, Pattern):
        return is_str(v) and value.search(v) is not None
    return v in as_list(value)


def and_(filters):
    def func(row, filters = filters):
        return min([_row_check(row, key, value) for key, value in filters.items()])
    return func
        

class dictable(Dict):
    """
    :What is dictable?:
    -------------------
    dictable is a table, a collection of iterable records. It is also a dict with each key being a column. 
    Why not use a pandas.DataFrame? pd.DataFrame leads a dual life: 
        - by day an index-based optimized numpy array supporting e.g. timeseries analytics etc.
        - by night, a table with keys supporting filtering, aggregating, pivoting on keys as well as inner/outer joining on keys.

    
    dictable only tries to do the latter. dictable should be thought of as a 'container for complicated objects' rather than just an array of primitive floats.
    In general, each cell may contain timeseries, yield_curves, machine-learning experiments etc.
    The interface is very succinct and allows the user to concentrate on logic of the calculations rather than boilerplate.
    
    dictable supports quite a flexible construction:

    :Example: construction using records
    ------------------------------------
    >>> from pyg import *; import pandas as pd
    >>> d = dictable([dict(name = 'alan', surname = 'atkins', age = 39, country = 'UK'), 
    >>>               dict(name = 'barbara', surname = 'brown', age = 29, country = 'UK')])
    
    :Example: construction using columns and constants
    ---------------------------------------------------
    >>> d = dictable(name = ['alan', 'barbara'], surname = ['atkins', 'brown'], age = [39, 29], country = 'UK')
    
    :Example: construction using pandas.DataFrame
    ---------------------------------------------
    >>> original = dictable(name = ['alan', 'barbara'], surname = ['atkins', 'brown'], age = [39, 29], country = 'UK')
    >>> df_from_dictable = pd.DataFrame(original)
    >>> dictable_from_df = dictable(df_from_dictable)
    >>> assert original == dictable_from_df

    :Example: construction rows and columns
    ---------------------------------------
    >>> d = dictable([['alan', 'atkins', 39, 'UK'], ['barbara', 'brown', 29, 'UK']], columns = ['name', 'surname', 'age', 'country'])


    :Access: column access
    ----------------------
    >>> assert d.keys() ==  ['name', 'surname', 'age', 'country']
    >>> assert d.name == ['alan', 'barbara']
    >>> assert d['name'] == ['alan', 'barbara']
    >>> assert d['name', 'surname'] == [('alan', 'atkins'), ('barbara', 'brown')]
    >>> assert d[lambda name, surname: '%s %s'%(name, surname)] == ['alan atkins', 'barbara brown']


    :Access: row access & iteration
    -------------------------------
    >>> assert d[0] == {'name': 'alan', 'surname': 'atkins', 'age': 39, 'country': 'UK'}
    >>> assert [row for row in d] == [{'name': 'alan', 'surname': 'atkins', 'age': 39, 'country': 'UK'},
    >>>                               {'name': 'barbara', 'surname': 'brown', 'age': 29, 'country': 'UK'}]

    Note that members access is commutative: 

    >>> assert d.name[0] == d[0].name == 'alan'
    >>> d[lambda name, surname: name + surname][0] == d[0][lambda name, surname: name + surname]
    >>> assert sum([row for row in d], dictable()) == d

    :Example: adding records
    ------------------------
    >>> d = dictable(name = ['alan', 'barbara'], surname = ['atkins', 'brown'], age = [39, 29], country = 'UK')
    >>> d = d + {'name': 'charlie', 'surname': 'chocolate', 'age': 49} # can add a record directly
    >>> assert d[-1] == {'name': 'charlie', 'surname': 'chocolate', 'age': 49, 'country': None}
    >>> d += dictable(name = ['dana', 'ender'], surname = ['deutch', 'esterhase'], age = [10, 20], country = ['Germany', 'Hungary'])
    >>> assert d.name == ['alan', 'barbara', 'charlie', 'dana', 'ender']
    >>> assert len(dictable.concat([d,d])) == len(d) * 2

    :Example: adding columns
    ------------------------
    >>> d = dictable(name = ['alan', 'barbara'], surname = ['atkins', 'brown'], age = [39, 29], country = 'UK')
    
    >>> ### all of the below are ways of adding columns ####
    >>> d.gender == ['m', 'f']
    >>> d = d(gender = ['m', 'f'])
    >>> d['gender'] == ['m', 'f']
    >>> d2 = dictable(gender = ['m', 'f'], profession = ['astronaut', 'barber'])
    >>> d = d(**d2)

    :Example: adding derived columns
    --------------------------------
    >>> d = dictable(name = ['alan', 'barbara'], surname = ['atkins', 'brown'], age = [39, 29], country = 'UK')
    >>> d = d(full_name = lambda name, surname: proper('%s %s'%(name, surname))) 
    >>> d['full_name'] = d[lambda name, surname: proper('%s %s'%(name, surname))]
    >>> assert d.full_name == ['Alan Atkins', 'Barbara Brown']

    :Example: dropping columns
    ---------------------------
    >>> d = dictable(name = ['alan', 'barbara'], surname = ['atkins', 'brown'], age = [39, 29], country = 'UK')
    >>> del d.country # in place
    >>> del d['age'] # in place
    >>> assert (d - 'name')[0] ==  {'surname': 'atkins'} and d[0] == {'name': 'alan', 'surname': 'atkins'}

    :Example: row selection, inc/exc
    --------------------------------------
    >>> d = dictable(name = ['alan', 'barbara'], surname = ['atkins', 'brown'], age = [39, 29], country = 'UK')
    >>> assert len(d.exc(name = 'alan')) == 1
    >>> assert len(d.exc(lambda age: age<30)) == 1 # can filter on *functions* of members, not just members.
    >>> assert d.inc(name = 'alan').surname == ['atkins']
    >>> assert d.inc(lambda age: age<30).name == ['barbara']
    >>> assert d.exc(lambda age: age<30).name == ['alan']

    dictable supports:
        - sort 
        - group-by/ungroup
        - list-by/ unlist
        - pivot/unpivot
        - inner join, outer join and xor

    Full details are below.
    """
    def __init__(self, data = None, columns = None, **kwargs):
        kwargs = {key :_value(value) for key, value in kwargs.items()}
        data_kwargs = {key: _value(value) for key, value in _data_columns_as_dict(data, columns).items()}
        kwargs.update(data_kwargs)
        if is_strs(columns) and (len(data_kwargs) == 0 or not is_str(columns)):
            kwargs = {key : kwargs.get(key, [None]) for key in columns} if len(kwargs)>0 else {key : [] for key in columns}
        n = lens(*kwargs.values())
        kwargs = {str(key) if is_int(key) else key : value * n if len(value)==1 else value for key, value in kwargs.items()}
        super(dictable, self).__init__(kwargs)
        
    _dict = Dict

    def __len__(self):
        return lens(*self.values())
    
    @property
    def shape(self):
        return (len(self), len(self.keys()))
    
    @property
    def columns(self):
        return self.keys()

    def get(self, key, default = None):
        if key in self:
            return self[key]
        else:
            return [default] * len(self)
        
    def __iter__(self):
        for row in zip(*self.values()):
            yield self._dict(zip(self.keys(), row))
            
    def update(self, other):
        for k, v in other.items():
            self[k] = v
    
    def __setitem__(self, key, value):
        n = len(self)
        value = _value(value)
        if len(value) == n or len(self.keys()) == 0:
            pass
        elif len(value) == 1:
            value = value * n 
        else:
            raise ValueError('cannot set item of length %s in table of length %s'%(len(value), n))
        if isinstance(key, tuple):
            for k,v in zipper(key, zipper(*value)):
                super(dictable, self).__setitem__(str(k) if is_int(k) else k, list(v))
        else:            
            super(dictable, self).__setitem__(str(key) if is_int(key) else key, list(value))
    
    def __getitem__(self, item):
        if is_arr(item) and len(item.shape) == 1:
            item = list(item)
        if isinstance(item, slice):
            return type(self)({key : value[item] for key, value in self.items()})
        if isinstance(item, (dict_keys, dict_values, range)):
            item = list(item)
        if isinstance(item , list):
            if len(item) == 0:
                return type(self)(data = [], columns = self.keys())
            elif is_strs(item):
                return type(self)(super(dictable, self).__getitem__(item))
            elif is_bools(item):
                res = type(self)([row for row, tf in zipper(list(self), item) if tf])
                return res if len(res) else type(self)([], self.keys())
            elif is_ints(item):
                values = list(zip(*self.values()))
                return type(self)(data = [values[i] for i in item], columns = self.keys())
            else:
                raise ValueError('We dont know how to understand this item %s'%item)
        elif is_int(item):
            return self._dict({key : value[item] for key, value in self.items()})
        elif item in self.keys():
            return super(dictable, self).__getitem__(item)
        elif is_tuple(item):
            return list(zip(*[self[i] for i in item]))
        elif callable(item):
            return self.apply(item)
        else:
            raise KeyError('item %s not found'%item)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        elif attr.startswith('_'):
            return super(dictable, self).__getattr__(attr)
        elif attr.startswith('find_'):
            key = attr[5:]
            if key not in self:
                raise KeyError('cannot find %s in dictable'%key)
            def f(*args, **kwargs):
                items = self.inc(*args, **kwargs)
                if len(items) == 0:
                    raise ValueError('no %s found'%key)
                item = items[key]
                if len(item)>1:
                    item = list(set(item))
                    if len(item)>1:
                        raise ValueError('multiple %s found %s'%(key, item))
                return item[0]
            f.__name__ = 'find_%s'%key
            return f
        else:
            raise AttributeError('%s not found'%attr)
 
    def __setattr__(self, attr, value):
        if attr.startswith('_'):
            super(dictable, self).__setattr__(attr, value)
        else:
            self.__setitem__(attr, value)

    def __delattr__(self, attr):
        if attr.startswith('_'):
            super(dictable, self).__delattr__(attr)
        else:
            super(dictable, self).__delitem__(attr)
    
    def if_else(self, condition, if_true, if_false, **default_params):
        """
        
        allows for evaluation of either if_true or if_false function

        Example
        -------
        >>> table['col'] = table.if_else(lambda col: col is None, if_true, 'col')

        is equivakent to

        >>> table = table.if_none(col = if_true)
        
        """
        tf = {True: if_true, False: if_false}
        return [row.apply(tf[bool(row[condition])], **default_params) for row in self]
        
    
    def inc(self, *functions, **filters):
        """
        performs a filter on what rows to include

        :Parameters:
        ----------------
        *functions : callables or a dict
            filters based on functions of each row
        **filters : value or list of values
            filters per each column

        :Returns:
        -------
        dictable
            table with rows that satisfy all conditions.
            
        
        :Example: filtering on keys
        -------
        >>> from pyg import *; import numpy as np
        >>> d = dictable(x = [1,2,3,np.nan], y = [0,4,3,5])
        >>> assert d.inc(x = np.nan) == dictable(x = np.nan, y = 5)            
        >>> assert d.inc(x = 1) == dictable(x = 1, y = 0)            
        >>> assert d.inc(x = [1,2]) == dictable(x = [1,2], y = [0,4]) 

        :Example: filtering on regex
        -------
        >>> import re
        >>> d = dictable(text = ['once', 'upon', 'a', 'time', 'in', 'the', 'west', 1, 2, 3])
        >>> assert d.inc(text = re.compile('o')) == dictable(text = ['once', 'upon'])
        >>> assert d.exc(text = re.compile('e')) == dictable(text = ['upon', 'a', 'in', 1, 2, 3])
        
        
        :Example: filtering on callables
        --------------
        >>> from pyg import *; import numpy as np
        >>> d = dictable(x = [1,2,3,np.nan], y = [0,4,3,5])
        >>> assert d.inc(lambda x,y: x>y) == dictable(x = 1, y = 0)

        """
        res = self.copy()
        if len(functions) + len(filters) == 0:
            return res
        functions = as_list(functions)
        for function in functions:
            if type(function) == dict:
                filters.update(function)
            else:
                f = kwargs_support(function)
                res = type(self)([row for row in res if f(**row)])
        for key, value in filters.items():
            if value is None:
                res = res[[r is None for r in res[key]]]
            elif is_nan(value):
                res = res[[is_nan(r) for r in res[key]]]
            elif isinstance(value, Pattern):
                res = res[[is_str(r) and value.search(r) is not None for r in res[key]]]                
            else:
                value = as_list(value)
                res = res[[r in value for r in res[key]]]
        if len(res) == 0:
            return type(self)([], self.keys())
        return res                

    def one_or_none(self, *functions, exc = None, find = None, **filters):
        """
        implements a sql-alchemy like one_or_none
        
        :examples: simple find of a unique row:
        ----------
        >>> rs = dictable(a = [1,2,3,4,5], b = [4,5,6,4,3], c = list('abcde'))
        >>> rs.one_or_none(a = 3)
        {'a': 3, 'b': 6, 'c': 'c'}

        :examples: simple fail with two rows with b == 4
        ----------
        >>> with pytest.raises(ValueError):
        >>>     rs.one_or_none(b = 4)    
          
        :examples: using exc to exclude a == 1 to give us unique entry, and also, get back 'c' column
        ----------
        >>> assert rs.one_or_none(b = 4, exc = dict(a = 1), find = 'c') == 'd'
                    
        """
        res = self.inc(*functions, **filters)
        if exc:
            res = res.exc(**exc)
        if len(res) > 1:
            raise ValueError(f'Found multiple rows with with {functions} {filters} and exclude {exc}')
        if len(res) == 0:
            return None
        res = res[0]
        if find:
            res = res[find]
        return res

    
    def exc(self, *functions, **filters):
        """
        performs a filter on what rows to exclude

        :Parameters:
        ----------------
        *functions : callables or a dict
            filters based on functions of each row
        **filters : value or list of values
            filters per each column

        :Returns:
        -------
        dictable
            table with rows that satisfy all conditions excluded.
            
        
        :Example: filtering on keys
        -------
        >>> from pyg import *; import numpy as np
        >>> d = dictable(x = [1,2,3,np.nan], y = [0,4,3,5])
        >>> assert d.exc(x = np.nan) == dictable(x = [1,2,3], y = [0,4,3])         
        >>> assert d.exc(x = 1) == dictable(x = [2,3,np.nan], y = [4,3,5])
        >>> assert d.exc(x = [1,2]) == dictable(x = [1,2], y = [0,4]) 
        >>> assert d.exc(x=1, y = [4,5]) == d ## not excluding anything since x = 1 only if y = 0
        >>> assert d.exc(x=[1,2], y = [4,5]) == d.exc(x=2) ## not excluding x = 1 since true only if y = 0


        :Example: filtering on callables
        --------------
        >>> from pyg import *; import numpy as np
        >>> d = dictable(x = [1,2,3,np.nan], y = [0,4,3,5])
        >>> assert d.exc(lambda x,y: x>y) == dictable(x = 1, y = 0)
        
        """
        res = self.copy()
        if len(functions) + len(filters) == 0:
            return res
        functions = as_list(functions)
        for function in functions:
            if type(function) == dict:
                filters.update(function)
            else:
                f = kwargs_support(function)
                res = type(self)([row for row in res if not f(**row)])
        if filters and len(res):
            include = and_(filters)
            res = res[[not include(row) for row in res]]            
        if len(res) == 0:
            return type(self)([], self.keys())
        return res                


    def if_none(self, none = None, **kwargs):
        """
        runs a column calculation if a column is not there, or if the existing value is considered missing.
        
        Example:
        --------
        >>> rs = dictable(a = [1,'hello'])
        >>> rs = rs(b = try_none(lambda a: a[0]))  ## fails for numerical value
        >>> rs

        >>> dictable[2 x 2]
        >>> a    |b   
        >>> 1    |None
        >>> hello|h   
        
        what shall we do with integer values for which b is None?
        
        >>> rs = rs.if_none(b = lambda a: a+3)
        >>> rs
        >>> dictable[2 x 2]
        >>> a    |b   
        >>> 1    |4
        >>> hello|h   


        Example:
        -------- 
        >>> rs = dictable(a = range(3))
        >>> rs = rs(b = try_nan(lambda a: 10/a))
        >>> rs = rs.if_none(np.nan, b = 'sorry no can do')

        >>> dictable[3 x 2]
        >>> a|b              
        >>> 0|sorry no can do
        >>> 1|10.0           
        >>> 2|5.0        
        
        """
        if none is None:
            check = lambda value: value is None
        elif is_nan(none):
            check = is_nan
        elif not callable(none):
            check = lambda value: value in as_list(none)
        else:
            check = none
        res = self
        for key, value in kwargs.items():
            if key not in res.keys(): ## revert back to simple calc
                res = res(**{key: value})
            else:
                if callable(value):
                    res[key] = [row[key] if not check(row[key]) else row.apply(value, **{'key': key}) for row in res]
                else:
                    res[key] = [row[key] if not check(row[key]) else value for row in res]                    
        return res

    
    def apply(self, function, **default_params):
        f = kwargs_support(function)
        if isinstance(function, dict):
            default_params = dict(function) | default_params
        return [f(**_dict_in_place_update(default_params,row)) for row in self] ## we update in place since all rows share same keys


    def do(self, function, *keys):
        """
        applies a function(s) on multiple keys at the same time

        :Parameters:
        ----------------
        function : callable or list of callables
            function to be applied per column
        *keys : string/list of strings
            list of columns to be applied. If missing, applied to all columns

        :Returns:
        -------
        res : dictable

        :Example:
        --------------
        >>> from pyg import *
        >>> d = dictable(name = ['adam', 'barbara', 'chris'], surname = ['atkins', 'brown', 'cohen'])
        >>> assert d.do(proper) == dictable(name = ['Adam', 'Barbara', 'Chris'], surname = ['Atkins', 'Brown', 'Cohen'])

        :Example: using another column in the calculation
        -------
        >>> from pyg import *
        >>> d = dictable(a = [1,2,3,4], b = [5,6,9,8], denominator = [10,20,30,40])
        >>> d = d.do(lambda value, denominator: value/denominator, 'a', 'b')
        >>> assert d == dictable(a = 0.1, b = [0.5,0.3,0.3,0.2], denominator = [10,20,30,40])
        
        :Example: do nothing if empty do list
        ---------
        >>> d = dictable(a = [1,2,3], b = 4)
        >>> assert d.do(lambda x: x **2 ) == dictable(a = [1,4,9], b = 16)
        >>> assert d.do(lambda x: x **2, []) == d
        
        """
        res = self.copy()
        if len(keys)  == 0:
            keys = self.keys()
        keys = as_list(keys)
        for key in keys:    
            for f in as_list(function):
                args = as_list(try_none(getargs)(f))
                res[key] = [f(row[key], **{k : v for k, v in row.items() if k in args[1:]}) for row in res]
        return res

    @classmethod
    def concat(cls, *others):
        """
        adds together multiple dictables. equivalent to sum(others, self) but a little faster

        :Parameters:
        ----------------
        *others : dictables
            records to be added to current table

        :Returns:
        -------
        merged : dictable
            sum of all records
            
        :Example:
        -------
        >>> from pyg import *
        >>> d1 = dictable(a = [1,2,3])
        >>> d2 = dictable(a = [4,5,6])
        >>> d3 = dictable(a = [7,8,9])
        
        >>> assert dictable.concat(d1,d2,d3) == dictable(a = range(1,10))
        >>> assert dictable.concat([d1,d2,d3]) == dictable(a = range(1,10))
        """
        others = as_list(others)
        others = [cls(other) if not isinstance(other, cls) else other for other in others]
        if len(others) == 0:
            return cls()
        elif len(others) == 1:
            return others[0]
        concated = dict_concat(others)
        merged = cls({key : sum(value, []) for key, value in concated.items()}) 
        return merged

    def __repr__(self):
        return 'dictable[%s x %s]'%self.shape + '\n%s'%self.__str__(3, 150)
    
    
    def __str__(self, max_rows = None, max_width = 150):
        return self.to_string(cat = max_rows, mid = '...%i rows...'%len(self), max_width=max_width)
    
    def to_string(self, colsep = '|', rowsep = '', cat = None, max_rows = 5, max_width = None, mid = '...', header = None, footer = None):
        """
        convrts dictable to string
        :Example:
        ---------
        >>> colsep = '|'; rowsep = ''; cat = None; max_rows = 5; max_width = None; mid = '...'; header = None; footer = None
        
        """
        if cat and len(self) > 2 * cat:
            box = {key: [_text_box(key)] + 
                        [_text_box('dictable[%s x %s]\n%s'%(len(v), len(v.keys()), v.keys()) if isinstance(v, dictable) else v, max_rows) for v in value[:cat]+value[-cat:]]
                    for key, value in self.items()}
        else:
            box = {key: [_text_box(key)] + 
                        [_text_box('dictable[%s x %s]\n%s'%(len(v), len(v.keys()), v.keys()) if isinstance(v, dictable) else v, max_rows) for v in value]
                    for key, value in self.items()}
    
        chars = {key : max([max([len(r) for r in v]) for v in value]) for key, value in box.items()}
        padded = {key : [[r + ' ' * (chars[key]-len(r)) for r in v] for v in value] for key, value in box.items()}
        def text(padded, keys = None):
            padded_ = padded if keys is None else {key: padded[key] for key in keys}
            rows = list(zip(*padded_.values()))
            empty_rows = [[' '*chars[key]] for key in padded_]
            ns = [max([len(v) for v in row]) for row in rows]
            if len(rowsep) == 1:
                sep_rows = [[rowsep*chars[key]] for key in padded_]
                res = ['\n'.join([colsep.join(v) for v in zip(*[r + e * (n-len(r)) + s for r,e,s in zip(row, empty_rows, sep_rows)])]) 
                       for row,n in zip(rows, ns)]
            else:
                res = ['\n'.join([colsep.join(v) for v in zip(*[r+e*(n-len(r)) for r, e in zip(row, empty_rows)])]) for row, n in zip(rows, ns)]
            if mid and cat and len(self)>2*cat:
                n = (len(res)+1)//2
                res = res[:n] + [mid] + res[n:]
            if rowsep == 'header':
                res = res[:1] + ['+'.join(['-'*chars[key] for key in padded_])] + res[1:]
            if header:
                res = [header] + res
            if footer:
                res = res + [footer]
            return '\n'.join(res)
        if max_width is None or max_width<=0:
            return text(padded)
        else:
            keys = list(padded.keys())
            res = []
            while len(keys)>0:
                i = 0
                width = 0
                while width<max_width and i<len(keys):
                    width+= chars[keys[i]]
                    i+=1
                res.append(text(padded, keys[:i]))
                keys = keys[i:]
            return '\n\n'.join(res)

    def sort(self, *by, **byval):
        """
        Sorts the table either using a key, list of keys or functions of members. Also allows sort on specific orders of the keys
        
        
        :Example:
        -------
        >>> import numpy as np
        >>> self = dictable(a = [_ for _ in 'abracadabra'], b=range(11), c = range(0,33,3))
        >>> self.d = list(np.array(self.c) % 11)
        >>> res = self.sort('a', 'd')
        >>> assert list(res.c) == list(range(11))
        
                
        >>> d = dictable(a = ['a', 1, 'c', 0, 'b', 2]).sort('a')        
        >>> res = d.sort('a','c')
        >>> print(res)
        >>> assert ''.join(res.a) == 'aaaaabbcdrr' and list(res.c) == [0,4,8,9,10] + [2,3] + [1] + [7] + [5,6]
        
        >>> d = d.sort(lambda b: b*3 % 11) ## sorting again by c but using a function
        >>> assert list(d.c) == list(range(11))
        
        :Example: Sorting based on specific values
        ---------
        >>> rs = dictable(key = list('abcdeffg'), gender = list('mmffmmff'))
        >>> rs = rs.sort(key = ['c', 'a', 'f', 'd'], gender = ['f', 'm'])
        
        key|gender
        ----------
        c  |f     
        a  |m     
        f  |f     <--- female f before male f
        f  |m     <---
        d  |f     
        g  |f     <--- g,b,e entries with no first key provided in sort, so move to bottom of list
        b  |m     
        e  |m    
        """
        if len(self) == 0:
            return self.copy()
        elif len(by):
            keys = self[by]            
        elif len(byval):
            dicts = {k : dict(zip(vals, range(len(vals)))) for k, vals in byval.items()}
            keys = [[d.get(row[k], len(d)) for k,d in dicts.items()] for row in self]            
        else:
            return self.copy()
        keys2id = list(zip(keys, range(len(self))))
        _, rows = zip(*sort(keys2id))
        return type(self)({key: [value[i] for i in rows] for key, value in self.items()})

    def __add__(self, other):
        if other is None or (is_num(other) and other == 0):
            return self
        return self.concat(self, other)

    def __and__(self, other):
        """
        equivalent 
        
        Example:
        --------
        >>> from pyg import * 
        >>> rs = dictable(a = 1, b = 2, c = [3,4])
        >>> assert rs & ['c', 'd'] == rs[['c']]

        """
        return self[self.columns & other]
            
    __radd__ = __add__
    
    def _listby(self, by):
        keys = self[by]
        keys2id = sort(list(zip(keys, range(len(self)))))
        prev = None
        res = []
        row = []
        for key, i in keys2id:
            if len(row) == 0 or key==prev:
                row.append(i)
            else:
                res.append((prev, row))
                row = [i]
            prev = key 
        res.append((prev, row))
        return zip(*res)

    def listby(self,*by):
        if len(self) == 0:
            return self.copy()
        if len(by) == 0:
            by = self.keys()
        by = as_tuple(by)
        if len(by) == 0:
            return type(self)({key: [value] for key, value in dict(self).items()})
        xs, ids = self._listby(by)
        rtn = type(self)(xs, by)
        rtn.update({k: [[self[k][i] for i in y] for y in ids] for k in self.keys() if k not in by})
        return rtn
    
    @classmethod
    def read_excel(cls, io, sheet_name = None, floats = None, ints = None, dates = None, no_none = None):
        if is_str(io):
            io = io.lower()
            if '!' in io and sheet_name is None: ## handling input like: ExcelFile.xlsx!SheetName
                io, sheet_name = io.split('!')
            if not '.xls' in io:
                io = io + '.xlsx'
            io = pd.ExcelFile(io)
        elif isinstance(io, Path):
            io = pd.ExcelFile(io)
        if sheet_name is None:
            sheet_name = 0
        df = io.parse(sheet_name)
        res = cls(df)
        res = res.relabel(lambda v: v.strip().replace(' ', '_').replace('"','').replace("'",''))
        res = res.do(nan2none)
        if floats:
            res = res.do(try_back(as_float), floats)
        if ints:
            res = res.do(try_back(int), ints)
        if dates:
            res = res.do(try_back(ndt), dates)
        if no_none:
            for col in as_list(no_none):
                res = res.exc(**{col : None})
        for col in [col for col in res.keys() if col.startswith('Unnamed:')]:
            if set(res[col]) == set([None]):
                res = res-col
        return res
    
    def unlist(self):
        """
        undoes listby...
        
        :Example:
        -------
        >>> x = dictable(a = [1,2,3,4], b= [1,0,1,0])
        >>> x.listby('b')
        
        dictable[2 x 2]
        b|a     
        0|[2, 4]
        1|[1, 3]
        
        
        >>> assert x.listby('b').unlist().sort('a') == x

        :Returns:
        -------
        dictable
            a dictable where all rows with list in them have been 'expanded'.

        """
        return self.concat([row for row in self]) if len(self) else self
    
    def groupby(self,*by, grp = 'grp'):
        """
        Similar to pandas groupby but returns a dictable of dictables with a new column 'grp' 
        
        :Example:
        -------
        >>> x = dictable(a = [1,2,3,4], b= [1,0,1,0])
        >>> res = x.groupby('b')
        >>> assert res.keys() == ['b', 'grp']
        >>> assert is_dictable(res[0].grp) and res[0].grp.keys() == ['a']

        :Parameters:
        ----------------
        *by : str or list of strings
        
            gr.
        grp : str, optional
            The name of the column for the dictables per each key. The default is 'grp'.

        :Returns:
        -------
        dictable
            A dictable containing the original keys and a dictable per unique key.

        """
        if len(self) == 0:
            return self.copy()
        if len(by) == 0:
            by = self.keys()
        by = as_tuple(by)
        if len(by) == 0:
            raise ValueError('cannot groupby on no keys, left with original dictable')
        elif len(by) == len(self.keys()):
            raise ValueError('cannot groupby on all keys... nothing left to group')
        xs,ys = self._listby(by)
        rtn = type(self)(xs, by)
        rtn[grp] = [type(self)({k: [self[k][i] for i in y] for k in self.keys() if k not in by}) for y in ys]
        return rtn

    def ungroup(self, grp = 'grp'):
        """
        Undoes groupby

        :Example:
        -------
        >>> x = dictable(a = [1,2,3,4], b= [1,0,1,0])
        >>> self = x.groupby('b')
        
        :Parameters:
        ----------------
        grp : str, optional
            column name where dictables are. The default is 'grp'.

        :Returns:
        -------
        dictable.

        """
        return self.concat([row.pop(grp)(**row.do(lambda v: [v])) for row in self])
        
    
    def join(self, other, lcols = None, rcols = None, mode = None):
        """
        
        Performs either an inner join or a cross join between two dictables
        
        :Parameters:
        ------------
        other: dictable-like
            a table to join with. 
        
        lcols: str/list of str/callables
            the values on which we join on LHS
        
        rcols: str/list of str/callables
            the values on which we join on RHS. 
            
        If rcols are None, use the same as lcols. 
        If both are None, will inner join on joint columns
        If both are empty lists, will cross join
        
        mode: int/str/callable
            If we have SAME COLUMN NAMES which are NOT in the join, we need to resolve them.
            By default we return both LHS values and RHS values as a tuple
            mode = 0/'left'/'lhs' : return the LHS value
            mode = 1/'right'/'rhs': returns RHS value
            mode = 'l|r': return lhs if not None, else rhs
            mode = 'r|l': return rhs if not None, else lhs
            mode = callable: apply that function
        

        :Example: inner join
        -------------------------------
        >>> from pyg import *
        >>> x = dictable(a = ['a','b','c','a']) 
        >>> y = dictable(a = ['a','y','z'])
        >>> assert x.join(y) == dictable(a = ['a', 'a'])

        :Example: outer join
        -------------------------------
        >>> from pyg import *
        >>> x = dictable(a = ['a','b']) 
        >>> y = dictable(b = ['x','y'])
        >>> assert x.join(y) == dictable(a = ['a', 'a', 'b', 'b'], b = ['x', 'y', 'x', 'y'])

    
        Example: joining mode
        --------------------
        >>> from pyg import * 
        >>> x = dictable(a = ['a', 'b', 'c', 'd'], b = [1,2,3,4])
        >>> y = dictable(a = ['a', 'b', 'c', 'd'], b = [10,20,30,40])
        
        >>> assert x.join(y, 'a').b == [(1, 10), (2, 20), (3, 30), (4, 40)]
        >>> assert x.join(y, 'a', mode = 'r').b == x.join(y, 'a', mode = 1).b == [10,20,30,40]
        >>> assert x.join(y, 'a', mode = 'l').b == x.join(y, 'a', mode = 0).b == [1,2,3,4]
        >>> assert x.join(y, 'a', mode = add_).b == [11,22,33,44]

        """
        mode = _modes.get(lower(mode), mode)
        _lcols = lcols
        if not isinstance(other, dictable):
            other = dictable(other)
        if lcols is None:
            lcols = self.keys() & other.keys()
        if rcols is None:
            rcols = lcols
        lcols = as_tuple(lcols); rcols = as_tuple(rcols)
        if len(lcols)!=len(rcols):
            raise ValueError('cannot inner join when cols on either side mismatch in length %s vs %s'%(lcols, rcols))
        elif _lcols is None:
            if len(lcols) > 0:
                _print_cols(*lcols)
        cols = []
        for lcol, rcol in zip(lcols, rcols):
            if is_str(lcol):
                cols.append(lcol)
            elif is_str(rcol):
                cols.append(rcol)
            else:
                raise ValueError('Cannot use a formula to inner join on both left and right %s %s'%(lcol, rcol))
        lkeys = self.keys() - cols
        rkeys = other.keys() - cols
        jkeys = lkeys & rkeys
        lkeys = lkeys - jkeys
        rkeys = rkeys - jkeys
        
        if len(cols):
            lxs, lids = self._listby(lcols)
            rxs, rids = other._listby(rcols)
            ls = len(lxs)
            rs = len(rxs)
            l = 0
            r = 0
            res = []
            while l<ls and r<rs:
                while l<ls and r<rs and cmp(lxs[l],rxs[r]) == -1:
                    l+=1
                while l<ls and r<rs and cmp(lxs[l],rxs[r]) == 1:
                    r+=1
                if l<ls and r<rs and lxs[l] == rxs[r]:
                    res.append((lxs[l], lids[l], rids[r]))
                    r+=1
                    l+=1
            if len(res) == 0:
                return type(self)([], cols + lkeys + rkeys + jkeys)
            xs, lids, rids = zip(*res)
            ns = [len(l)*len(r) for l, r in zip(lids, rids)]
            rtn = type(self)(sum([[x]*n for x,n in zip(xs,ns)], []), cols)
        else:
            rtn = type(self)()
            lids = [range(len(self))]; rids = [range(len(other))]
        for k in lkeys:
            v= self[k]
            rtn[k] = sum([[v[l] for l in lid for r in rid] for lid, rid in zip(lids, rids)], [])
        for k in rkeys:
            v= other[k]
            rtn[k] = sum([[v[r] for l in lid for r in rid] for lid, rid in zip(lids, rids)], [])            
        for k in jkeys:
            if callable(mode):
                lv = self[k]
                rv = other[k]
                rtn[k] = sum([[mode(lv[l], rv[r]) for l in lid for r in rid] for lid, rid in zip(lids, rids)], [])            
            else:                
                lv = self[k]
                rv = other[k]
                rtn[k] = sum([[(lv[l], rv[r]) for l in lid for r in rid] for lid, rid in zip(lids, rids)], [])            
        return rtn
    
    def xor(self, other, lcols = None, rcols = None, mode = 'l'):
        """
        returns what is in lhs but NOT in rhs (or vice versa if mode = 'r'). Together with inner joining, can be used as left/right join

        :Examples:
        --------------
        >>> from pyg import *
        >>> self = dictable(a = [1,2,3,4])
        >>> other = dictable(a = [1,2,3,5])
        >>> assert self.xor(other) == dictable(a = 4) # this is in lhs but not in rhs
        >>> assert self.xor(other, lcols = lambda a: a * 2, rcols = 'a') == dictable(a = [2,3,4]) # fit can be done using formulae rather than actual columns

        The XOR functionality can be performed using quotient (divide):
        >>> assert lhs/rhs == dictable(a = 4)
        >>> assert rhs/lhs == dictable(a = 5)

        >>> rhs = dictable(a = [1,2], b = [3,4])
        >>> left_join_can_be_done_simply_as = lhs * rhs + lhs/rhs


        :Parameters:
        ----------------
        other : dictable (or something that can be turned to one)
            what we exclude with.
        lcols : str/list of strs, optional
            the left columns/formulae on which we match. The default is None.
        rcols : str/list of strs, optional
            the right columns/formulae on which we match. The default is None.
        mode : string, optional
            When set to 'r', performs xor the other way. The default is 'l'.

        :Returns:
        -------
        dictable
            a dictable containing what is in self but not in ther other dictable.

        """
        if not isinstance(other, dictable):
            other = dictable(other)
        if lcols is None:
            lcols = self.keys() & other.keys()
        if rcols is None:
            rcols = lcols
        lcols = as_tuple(lcols); rcols = as_tuple(rcols)
        if len(lcols)!=len(rcols):
            raise ValueError('cannot xor-join when cols on either side mismatch in length %s vs %s'%(lcols, rcols))
        if len(lcols) == 0:
            return self.copy()
        lxs, lids = self._listby(lcols)
        rxs, rids = other._listby(rcols)
        mode = 1 if (is_str(mode) and mode[0].lower() == 'r') or mode == 1 else 0
        ls = len(lxs)
        rs = len(rxs)
        l = 0
        r = 0
        res = []
        while l<ls and r<rs:
            while l<ls and r<rs and cmp(lxs[l],rxs[r]) == -1:
                if mode == 0:
                    res.append(lids[l])
                l+=1
            while l<ls and r<rs and cmp(lxs[l],rxs[r]) == 1:
                if mode == 1:
                    res.append(rids[r])                    
                r+=1
            if l<ls and r<rs and lxs[l] == rxs[r]:
                r+=1
                l+=1
        if mode == 0:
            if l<ls:
                res.extend(lids[l:])
            return self[sum(res, [])]
        else:
            if r<rs:
                res.extend(rids[r:])
            return other[sum(res, [])]

    __mul__ = join
    __truediv__ = xor
    __div__ = xor
    
    def left_join(self, other, lcols = None, rcols = None, mode = None, **defaults):
        res = self.join(other, lcols = lcols, rcols = rcols, mode = mode) + self.xor(other, lcos = lcols, rcols = rcols, mode = mode)
        return res.if_none(**defaults)

    def right_join(self, other, lcols = None, rcols = None, mode = None, **defaults):
        res = self.join(other, lcols = lcols, rcols = rcols, mode = mode) + dictable(other).xor(self, lcos = rcols, rcols = lcols, mode = mode)
        return res.if_none(**defaults)
    
    def xyz(self, x, y, z, agg = None):
        """
        
        pivot table functionality.
        
        :Parameters:
        ----------------
        x : str/list of str
            unique keys per each row
        y : str
            unique key per each column
        z : str/callable
            A column in the table or an evaluated quantity per each row
        agg : None/callable or list of callables, optional
            Each (x,y) cell can potentially contain multiple z values. so if agg = None, a list is returned
            If you want the data aggregated in any way, then supply an aggregating function(s)

        :Returns:
        -------
        A dictable which is a pivot table of the original data
        

        :Example:
        -------
        >>> from pyg import *
        >>> timetable_as_list = dictable(x = [1,2,3]) * dictable(y = [1,2,3]) 
        >>> timetable = timetable_as_list.xyz('x','y',lambda x, y: x * y)
        >>> assert timetable = dictable(x = [1,2,3], )

        :Example:
        -------
        >>> self = dictable(a = 1, b = 2 , c = 'c', d = 3)
        >>> x = 'x'; y = 'y'; z = lambda x, y: x * y
        >>> self.exc(lambda x, y: x+y==5).xyz(x,y,z, len)
        
        """
        if not is_strs(x):
            raise ValueError('x must be columns %s'%x)
        agg = as_list(agg)
        x = as_tuple(x)
        xykeys = x + as_tuple(y)
        xys, ids = self._listby(xykeys)
        zs = self[z]
        y_ = y if is_str(y) else '_columns'
        rs = type(self)(xys, x + (y_,))        
        ys = rs[as_list(y_)].listby(y_)
        y2id = dict(zip(ys[y_], range(len(ys))))
        xs, yids = rs._listby(x)
        res = [[None for _ in range(len(ys))] for _ in range(len(xs))]
        for i in range(len(xs)):
            for j in yids[i]:
                xy = xys[j]
                k = y2id[xy[-1]]
                value = [zs[id_] for id_ in ids[j]]
                if agg:
                    for a in agg:
                        value = a(value)
                res[i][k] = value
        dx = type(self)(xs, x)
        dy = type(self)(res, list(y2id.keys()))
        dx.update(dy)
        return dx
    
    pivot = xyz
    
    def unpivot(self, x, y, z):
        """
        undoes self.xyz / self.pivot
        
        :Example:
        -------
        >>> from pyg import *
        >>> orig = (dictable(x = [1,2,3,4]) * dict(y = [1,2,3,4,5]))(z = lambda x, y: x*y)
        >>> pivot = orig.xyz('x', 'y', 'z', last)
        >>> unpivot = pivot.unpivot('x','y','z').do(int, 'y') # the conversion to column names mean y is now string... so we convert back to int
        >>> assert orig == unpivot

        :Parameters:
        ----------------
        x : str/list of strings
            list of keys in the pivot table.
        y : str
            name of the columns that wil be used for the values that are currently column headers.
        z : str
            name of the column that describes the data currently within the pivot table.

        :Returns:
        -------
        dictable

        """
        xcols = as_list(x)
        if isinstance(y, dict) and len(y) == 1:
            y, ycols = list(y.items())[0]
        else:
            ycols = self.keys() - x
        ycols = as_tuple(ycols)
        n = len(ycols)

        res = type(self)({k: sum([[row[k]]*n for row in self], []) for k in xcols})
        res[y] = ycols * len(self)
        res[z] = sum([[row[ycol] for ycol in ycols] for row in self], [])
        return res
    

            

def is_dictable(value):
    return isinstance(value, dictable)

