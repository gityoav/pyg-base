from pyg_base._dict import loop
from pyg_base._types import is_str, is_num

_endings = [('million',6), ('billion', 9), ('trillion', 12), ('percent', -2), ('mln', 6), ('bln', 9), ('tln', 12), ('trl', 12), ('pct', -2), ('mn', 6), ('bn', 9), ('tn', 12), ('bp', -4), ('%', -2), ('m', 6), ('k', 3), ('b', 9), ('t', 12), ('crore', 7), ('lakh', 5)]
_n = [(k, 10**v) for k, v in _endings]

__all__ = ['as_float']

def _txt_and_mult(txt):
    """
    """
    lower_txt = txt.lower()
    if len(txt) < 2 or txt[-1].isdigit():
        return txt, 1
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
    >>> assert as_float('1.4 mln') == 1400000
    >>> assert as_float('1.4bn') == 1400000000
    >>> assert as_float('1.4tln') == 1400000000000
    >>> assert as_float('100%') == 1
    >>> assert as_float('100 pct') == 1
    >>> assert as_float('100bp') == 0.01
    >>> assert as_float('1,234') == 1234
    >>> assert as_float('-1,234k') == -1234000
    >>> assert as_float('1.2 lakh') == 120000 ## indian rupees notations
    >>> assert as_float('1.2 crore') == 12000000 ## indian rupees notations
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