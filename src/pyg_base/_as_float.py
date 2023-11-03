from pyg_base._loop import loop
from pyg_base._types import is_str, is_num

_endings = [('mln', 6), ('mn', 6), ('m', 6), ('bln', 9), ('bn', 9), ('b', 9), ('tln', 12), ('tn', 12), ('t', 12), ('%', -2), ('k', 3)]
_n = [(k, 10**v) for k, v in _endings]

__all__ = ['as_float']

def _txt_and_mult(txt):
    """
    """
    lower_txt = txt.lower()
    for k, mult in _n:
        if lower_txt.endswith(k):
            return txt[:-len(k)], mult
    return txt, 1


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
    >>> assert as_float('1.4mln') == 1400000
    >>> assert as_float('1.4bn') == 1400000000
    >>> assert as_float('1.4tln') == 1400000000000
    >>> assert as_float('100%') == 1
    >>> assert as_float('1,234') == 1234
    >>> assert as_float('-1,234k') == -1234000
    """
    if is_str(value):
        txt = value.replace(',','').replace(' ','')
        if not txt:
            return None
        txt, mult = _txt_and_mult(txt)
        try:
            res = float(txt)
            res = round(res, len(txt)+2)
            return res * mult
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