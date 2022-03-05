from pyg_base._loop import loop
from pyg_base._types import is_str, is_num

_k = {0 : '', 3 : 'k', 6:'m', 9:'b', 12: 't', -2 : '%'}
_n = {v: 10**k for k,v in _k.items()}
_n.update({v.upper(): 10**k for k,v in _k.items()})

__all__ = ['as_float']

@loop(list, tuple, dict)
def _as_float(value):
    """
    converts a string to float

    :Parameters:
    ----------
    value : string
        number in string format.

    :Returns:
    -------
    float

    :Example:
    --------
    >>> from pyg import *
    >>> assert as_float('1.3k') == 1300
    >>> assert as_float('1.4m') == 1400000
    >>> assert as_float('100%') == 1
    >>> assert as_float('1,234') == 1234
    >>> assert as_float('-1,234k') == -1234000
    """
    if is_str(value):
        txt = value.replace(',','').replace(' ','')
        if not txt:
            return None
        if txt[-1] in _n:
            mult = _n[txt[-1]]
            txt = txt[:-1]
        else:
            mult = 1
        try:
            res = mult * float(txt)
            res = round(res, len(txt)+2)
            return res
        except ValueError:
            return value
    else:
        return value

def as_float(value):
    """
    converts a string to float allowing for commas, percentages etc.

    :Parameters:
    ----------
    value : string
        number in string format.

    :Returns:
    -------
    float

    :Example:
    --------
    >>> from pyg import *
    >>> assert as_float('1.3k') == 1300
    >>> assert as_float('1.4m') == 1400000
    >>> assert as_float('100%') == 1
    >>> assert as_float('1,234') == 1234
    >>> assert as_float('-1,234k') == -1234000
    """
    return _as_float(value)