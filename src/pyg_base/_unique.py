from pyg_base._as_list import as_list
from pyg_base._eq import eq


def first(value):
    """
    returns the first value in a list (None if empty list) or the original if value not a list

    :Example:
    ---------
    >>> assert first(5) == 5
    >>> assert first([5,5]) == 5
    >>> assert first([]) is None
    >>> assert first([1,2]) == 1
    
    """
    values = as_list(value)
    return values[0] if len(values) else None

def last(value):
    """
    returns the last value in a list (None if empty list) or the original if value not a list

    :Example:
    ---------
    >>> assert last(5) == 5
    >>> assert last([5,5]) == 5
    >>> assert last([]) is None
    >>> assert last([1,2]) == 2
    
    """
    values = as_list(value)
    return values[-1] if len(values) else None

def unique(value):
    """
    returns the asserted unique value in a list (None if empty list) or the original if value not a list. 
    Throws an exception if list non-unique
    
    :Example:
    ---------
    >>> assert unique(5) == 5
    >>> assert unique([5,5]) == 5
    >>> assert unique([]) is None
    >>> with pytest.raises(ValueError):
    >>>     unique([1,2])  
    
    """
    values = as_list(value)
    if len(values) == 0:
        return None
    elif len(values) == 1:
        return values[0]
    else:
        res = values[0]
        try:
            if len(set(values)) == 1:
                return res
            else:
                raise ValueError('values provided not unique %s'%values)
        except TypeError:
            for v in values[1:]:
                if not eq(v, res):
                    raise ValueError('values provided not unique %s, %s '%(res,v))
            return res                
        
# -*- coding: utf-8 -*-

