from _collections_abc import dict_keys, dict_values

def is_rng(value):
    return isinstance(value, (list, tuple, range, set, dict_keys, dict_values, zip) )


__all__ = ['as_list', 'as_tuple', 'is_rng', 'passthru']

def as_list(value, none = False):
    """
    returns a list of the original object. 
 
    :Example:
    ---------
    >>> assert as_list(None) == []
    >>> assert as_list(4) == [4]
    >>> assert as_list((1,2,)) == [1,2]
    >>> assert as_list([1,2,]) == [1,2]
    >>> assert eq(as_list(np.array([1,2,])) , [np.array([1,2,])])
    >>> assert as_list(dict(a = 1)) == [dict(a=1)]
 
    In practice, this function is has an incredible useful usage:

    :Example: using as_list to give flexibility on *args
    ----------------------------------------------------
    >>> def my_sum(*values):
    >>>     values = as_list(values)
    >>>     return sum(values)

    >>> assert my_sum(1,2,3) == 6    
    >>> assert my_sum([1,2,3]) == 6 ## This is nice... wasn't possible before
    
    :Parameters:
    ----------------
    value : anything
    none : bool optional
        Shall I return None as a value? The default is False and we return [], if True, returns [None]

    :Returns:
    -------
    list
        a list of original objects.

    """
    if value is None and not none:
        return []
    elif isinstance(value, list):
        return value
    elif isinstance(value, tuple):
        if len(value)==1 and isinstance(value[0], list):
            return value[0]
        else:
            return list(value)
    elif is_rng(value):
        return list(value)
    else:
        return [value]

def as_tuple(value, none = False):
    """
    returns a tuple of the original object. 
 
    :Example:
    ---------
    >>> assert as_tuple(None) == ()
    >>> assert as_tuple(4) == (4,)
    >>> assert as_tuple((1,2,)) == (1,2)
    >>> assert as_tuple([1,2,]) == (1,2)
    >>> assert eq(as_tuple(np.array([1,2,])) , (np.array([1,2,]),))
    >>> assert as_tuple(dict(a = 1)) == (dict(a=1),)
 
    In practice, this function is has an incredible useful usage:

    :Example: using as_list to give flexibility on *args
    ----------------------------------------------------
    >>> def my_sum(*values):
    >>>     values = as_tuple(values)
    >>>     return sum(values)

    >>> assert my_sum(1,2,3) == 6    
    >>> assert my_sum([1,2,3]) == 6 ## This is nice... wasn't possible before
    
    :Parameters:
    ----------------
    value : anything
    none : bool optional
        Shall I return None as a value? The default is False and we return [], if True, returns [None]

    :Returns:
    -------
    tuple
        a tuple of original objects.
    """
    if value is None and not none:
        return ()
    elif isinstance(value, tuple):
        if len(value)==1 and isinstance(value[0], list):
            return tuple(value[0])
        else:
            return value
    elif is_rng(value):
        return tuple(value)
    else:
        return (value,)


def passthru(data):
    """
    does nothing. returns data
    """
    return data