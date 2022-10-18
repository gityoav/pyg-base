from pyg_base._types import is_int, is_float, is_str, is_num, is_nan, is_ts, is_date
from pyg_base._as_list import as_list
from pyg_base._cache import cache
import datetime
import re
import pandas as pd
import numpy as np
from dateutil import parser
# from dateutil.relativedelta import relativedelta
from functools import reduce, partial
import dateutil as du
from dateutil import tz
from pytz import country_timezones
import pytz


NaT = pd.NaT
NaTType = type(NaT)
TMIN = datetime.datetime(1900,1,1)
TMAX = datetime.datetime(2300,1,1)
microsecond = datetime.timedelta(microseconds = 1)
DAY = datetime.timedelta(days = 1)
iso = re.compile('^[0-9]{4}-[0-9]{2}-[0-9]{2}T')
ambiguity = re.compile('^[0-9]{1,2}[-/ .][0-9]{1,2}[-/ .][0-9]{2,4}')
futcodes = list('fghjkmnquvxz'.upper())
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
yyyymm = re.compile('^[0-9]{4}[-/ .][0-9]{1,2}$')
yyyymmm = re.compile('^[0-9]{4}[-/ .](%s)[a-z]*$'%('|'.join(months)), re.IGNORECASE)
wkdays = dict(mon = 0, tue = 1, wed = 2, thu = 3, fri = 4, sat = 5, sun = 6)
period = re.compile('^[-+]{0,1}[0-9]+[dbwmqyhnsDBWMQYHNS]{1}')
mmm2m = dict(jan = 1, feb = 2, mar = 3, apr = 4, may = 5, jun = 6, jul = 7, aug = 8, sep = 9, oct = 10, nov = 11, dec = 12)

__all__ = ['dt','dt_bump', 'today', 'ymd', 'TMIN', 'TMAX', 'DAY', 'futcodes', 'dt2str', 'is_period', 'nth_weekday_of_month']

@cache
def tzones():
    x = {iso :country_timezones[iso] for iso in country_timezones}
    tzs = {iso: code[0] for iso, code in x.items() if len(code) == 1}
    for iso, code in x.items():
        tzs.update({c.split('/')[-1].replace('_', ' ').lower(): c for c in code})
    tzs  = {key : tz.gettz(value) for key, value in tzs.items()}
    t = datetime.datetime.now()
    tzs.update({v.tzname(t): v for v in tzs.values() if v is not None})
    tzs['WET'] = tzs['western european'] = tzs['MA'] # Malta
    tzs['EET'] = tzs['eastern european'] = tzs['GR']
    tzs['CET'] = tzs['cental european'] = tzs['AT'] # austria
    tzs['EST'] = tzs['eastern standard'] = tzs['new york']
    tzs['CST'] = tzs['central standard'] = tzs['chicago']
    tzs['greenwich mean time'] = tzs['GMT']
    return tzs

def as_tz(tzinfo):
    if is_str(tzinfo):
        zones = tzones()
        if tzinfo in zones:
            return zones[tzinfo]
        elif tzinfo.lower() in zones:
            return zones[tzinfo.lower()]
        elif tzinfo.upper() in zones:
            return zones[tzinfo.upper()]
        else:
            return pytz.timezone(tzinfo)
    else:
        return tzinfo

def tz_convert(t, tzinfo = None):
    """
    if tzinfo is None: drops all timezone data
    if tzinfo is not None: converts/localize the existing date to 
    

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    tzinfo : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if tzinfo is None:
        ## we remove tzinfo
        if isinstance(t, (pd.DataFrame, pd.Series)):
            if t.index.tz is None:
                return t
            else:
                res = t.copy()
                res.index = [date.replace(tzinfo = None) for date in res.index]
                return res
        elif is_date(t):
            return t if t.tzinfo is None else t.replace(tzinfo = None)
        elif isinstance(t, list):
            return type(t)([tz_convert(i, tzinfo) for i in t])
        elif isinstance(t, dict):
            return type(t)({key : tz_convert(value, tzinfo) for key, value in t.items()})
        else:
            return dt(t, tzinfo = tzinfo)
    if isinstance(t, list):
        return type(t)([tz_convert(i, tzinfo) for i in t])
    elif isinstance(t, dict):
        return type(t)({key : tz_convert(value, tzinfo) for key, value in t.items()})
    if isinstance(t, (pd.DataFrame, pd.Series)):
        res = t.copy()
        if t.index.tz is None:
            res.index = res.index.tz_localize(as_tz(tzinfo))
        else:
            res.index = res.index.tz_convert(as_tz(tzinfo))
        return res
    elif is_date(t):
        return t.astimezone(as_tz(tzinfo))
    else:
        return dt(t, tzinfo)

def today(date = None):
    now = date or datetime.datetime.now()
    return NaT if isinstance(date, NaTType) else datetime.datetime(now.year, now.month, now.day)


def month(m):
    """
    converts m into a valid month
    
    :Examples:
    ---------------
    >>> assert month(3) == 3
    >>> assert month('h') == 3
    >>> assert month('march') == 3

    :Parameters:
    ----------------
    m : str/int
        DESCRIPTION.    

    Raises
    ------
    ValueError
        DESCRIPTION.

    :Returns:
    -------
    int
        month in integer 1...12

    """
    if is_int(m):
        return m
    elif is_float(m) and int(m) == m:
        return int(m)
    elif is_str(m):
        if len(m) == 1 :
            return futcodes.index(m.upper()) + 1
        else:
            return months.index(m[:3].lower()) + 1
    else:
        raise ValueError('do not understand month %s' % m)

def ym(y,m):
    """
    converts a y,m into actual year-month
    
    :Example:
    --------------
    >>> assert ym(2000, -1) == (1999,11)
    >>> assert ym(2000, 0) == (1999,12)
    >>> assert ym(2000, 1) == (2000,1)
    >>> assert ym(2000, 'h') == (2000,3)
    >>> assert ym(2000, 'March') == (2000,3)
    ...
    >>> assert ym(2000, 12) == (2000,12)
    >>> assert ym(2000, 13) == (2001,1)
    
    """
    y = int(y) if is_float(y) and int(y) == y else y
    m = month(m)
    y += (m-1) // 12
    m = 1 + ((m-1) % 12)
    return (y, m)

def _ymd(y,m,d):
    """
    returns a date from year, month, day.
    
    WARNING: Handles date arithmetic not quite how you expect it

    :Parameters:
    ----------------
    y : int
        year.
    m : int
        month offsets from 0.
    d : int
        day offsets from 0 

    :Returns:
    -------
    date
        calculates dates.

    """
    if d>1500 and d<3000 and y>0 and y<32 and m>0 and m<13:
        y,d = d,y
    y,m = ym(y,m)
    d = int(d) if is_float(d) and int(d) == d else d
    return datetime.datetime(y,m,1) + (d-1) * DAY

def num2dt(n):
    i = int(n); f = datetime.timedelta(n - i)
    if i<=1500:
        return today() + i * DAY + f
    elif i<=3000:
        return datetime.datetime(i, 1, 1) + f
    elif i < 300000:
        return datetime.datetime.fromordinal(i + 693594) + f
    elif i<1095000:
        return datetime.datetime.fromordinal(i) + f
    elif i>10000101 and i<30001231:
        y = i // 10000
        m = (i % 10000) // 100
        d = i % 100
        return _ymd(y,m,d) + f
    else:
        return datetime.datetime.utcfromtimestamp(n)

# def int2dt(i):
#     if i<=1500:
#         return today() + i * day
#     elif i<=3000:
#         return datetime.datetime(i, 1, 1)
#     elif i < 300000:
#         return datetime.datetime.fromordinal(i + 693594)
#     elif i<1095000:
#         return datetime.datetime.fromordinal(i)
#     elif i>10000101 and i<30001231:
#         y = i // 10000
#         m = (i % 10000) // 100
#         d = i % 100
#         return ymd(y,m,d)
#     else:
#         return datetime.datetime.utcfromtimestamp(i)
    
def np2dt(t):
    """
    >>> d = datetime.datetime(2000,1,1,20,30,40,55)
    >>> t = np.datetime64(d)    
    >>> assert dt(t) == d
    >>> t = np.datetime64(d).astype('datetime64[D]')
    >>> assert np2dt(t) == dt(d)
    >>> t = np.datetime64(d).astype('datetime64[m]')
    >>> assert dt(t) == datetime.datetime(2000,1,1,20,30)
    >>> t = np.datetime64(d).astype('datetime64[ms]')
    >>> assert dt(t) == datetime.datetime(2000,1,1,20,30,40)
    >>> t = np.datetime64(d).astype('datetime64[ns]')
    >>> assert dt(t) == d
    
    Parameters
    ----------
    t : numpy.datetime64 format
        time.

    Returns
    -------
    datetime
        datetime object.
    """
    
    res = t.astype(datetime.datetime)
    if isinstance(res, datetime.datetime):
        return res
    elif isinstance(res, datetime.date): # [D] format
        return datetime.datetime(res.year, res.month, res.day)
    elif is_int(res): ## [ns] format
        return pd.Timestamp(t)
    return res

        
def uk2dt(t, tzinfo = None):
    if t in ('', 'null'):
        return None
    elif t in ('NaT'):
        return NaT
    elif t.lower() == 'now':
        return datetime.datetime.now()
    res = parser.parse(t)
    if ambiguity.search(t) is not None:
        if res.day<13:
            res = dt(res.year, res.day, res.month, res.hour, res.minute, res.second, res.microsecond)
        elif int(t[:2].replace('-','').replace('/',''))!=res.day:
            raise ValueError('date %s is not in UK date format'%t)
    elif yyyymm.search(t) is not None or yyyymmm.search(t) is not None:
        res = datetime.datetime(res.year, res.month, 1)
    return tz_convert(res, tzinfo)


def us2dt(t, tzinfo = None):
    if t in ('', 'null'):
        return None
    elif t in ('NaT'):
        return NaT
    elif t.lower() == 'now':
        return datetime.datetime.now()
    res = parser.parse(t)
    if ambiguity.search(t) is not None and res.month != int(t[:2].replace('-','').replace('/','')):
        raise ValueError('the date is not in US format')
    if yyyymm.search(t) is not None or yyyymmm.search(t) is not None:
        res = datetime.datetime(res.year, res.month, 1)
    return tz_convert(res, tzinfo)

def none2dt(none = datetime.datetime.now):
    if callable(none):
        return none()
    else:
        return none

def is_period(bump):
    return is_str(bump) and period.search(bump) is not None    

def is_bump(bump):
    return is_period(bump) or (is_int(bump) and bump<1500) or isinstance(bump, (datetime.timedelta, du.relativedelta.relativedelta))

def dt_bump(t, *bumps):
    """
    :Example:
    ---------
    >>> from pyg import *
    >>> t  = pd.Series([1,2,3], drange(dt(2000,1,1),2))
    >>> assert eq(dt_bump(t, 1), pd.Series([1,2,3], drange(dt(2000,1,2),2)))

    Example: multiple tenors
    ------------------------
    >>> t = dt(2000)
    >>> assert dt_bump(t, '1y1m1d') == dt(2001,2,2) ## move up a year a month and a day
    >>> assert dt_bump(t, '1y1m-1d') == dt(2001,1,31) ## move up a year and a month, then a day back

    """
    bumps = as_list(bumps)
    if is_ts(t):
        res = t.copy()
        res.index = [dt_bump(i, *bumps) for i in res.index]
        return res
    t = t if isinstance(t, datetime.datetime) else dt(t)
    if isinstance(t, NaTType):
        return NaT
    for bump in bumps:
        if is_int(bump):
            t = t + DAY * bump
        elif isinstance(bump, (datetime.timedelta, du.relativedelta.relativedelta)):
            t = t + bump
        elif is_str(bump):
            bump = bump.lower()
            while period.search(bump) is not None:
                bmp = period.search(bump).group()
                bump = bump[len(bmp):]
                if bmp.endswith('d'):
                    t = t + DAY * int(bmp[:-1])
                elif bmp.endswith('w'):
                    t  = t + DAY * (7 * int(bmp[:-1]))
                elif bmp.endswith('m'):
                    t = _ymd(t.year, t.month + int(bmp[:-1]), t.day)
                elif bmp.endswith('q'):
                    t = _ymd(t.year, t.month + 3  * int(bmp[:-1]), t.day)
                elif bmp.endswith('y'):
                    t = _ymd(t.year+int(bmp[:-1]), t.month, t.day)
                elif bmp.endswith('h'):
                    t = t + datetime.timedelta(hours = int(bmp[:-1]))
                elif bmp.endswith('n'):
                    t = t + datetime.timedelta(minutes = int(bmp[:-1]))
                elif bmp.endswith('s'):
                    t = t + datetime.timedelta(seconds = int(bmp[:-1]))
                elif bmp.endswith('b'):
                    bdays = int(bmp[:-1])
                    wday = t.weekday()
                    if wday>4:
                        t = t + (7-wday) * DAY
                        wday = 0
                    w = bdays // 5
                    d = bdays - w * 5
                    t = t + DAY * (7*w)
                    if wday + d > 4:
                        d+=2
                    t += DAY * d
            if len(bump):
                try:
                    t = tz_convert(t, bump)
                except Exception:
                    raise ValueError('%s is not a period I know...'%bump)
        else:
            t = t + bump
    return t

def dt(*args, dialect = 'uk', none = datetime.datetime.now, tzinfo = None):
    """
    A more generic constructor for datetime.datetime. 
    
    :Example: Simple construction
    --------------
    >>> assert dt(2000,1 ,1) == datetime.datetime(2000, 1, 1, 0, 0) # name of month
    >>> assert dt(2000,'jan',1) == datetime.datetime(2000, 1, 1, 0, 0) # name of month
    >>> assert dt(2000,'f',1) == datetime.datetime(2000, 1, 1, 0, 0) # future month code
    >>> assert dt('01-02-2002') == datetime.datetime(2002, 2, 1)
    >>> assert dt('01-02-2002', dialect = 'US') == datetime.datetime(2002, 1, 2)
    >>> assert dt('01 March 2002') == datetime.datetime(2002, 3, 1)
    >>> assert dt('01 March 2002', dialect = 'US') == datetime.datetime(2002, 3, 1)
    >>> assert dt('01 March 2002 10:20:30') == datetime.datetime(2002, 3, 1, 10, 20, 30)

    >>> assert dt(20020301) == datetime.datetime(2002, 3, 1)
    >>> assert dt(37316) == datetime.datetime(2002, 3, 1) # excel date
    >>> assert dt(730180) == datetime.datetime(2000,3,1) # ordinal for 1/3/2000
    >>> assert dt(2000,3,1).timestamp() == 951868800.0
    >>> assert dt(951868800.0) == datetime.datetime(2000,3,1) # utc timestamp
    >>> assert dt(np.datetime64(dt(2000,3,1))) == dt(2000,3,1) ## numpy.datetime64 object

    >>> assert dt(2000) == datetime.datetime(2000,1,1)
    >>> assert dt(2000,3) == datetime.datetime(2000,3,1)
    >>> assert dt(2000,3, 1) == datetime.datetime(2000,3,1)
    >>> assert dt(2000,3, 1, 10,20,30) == datetime.datetime(2000,3,1,10,20,30)
    >>> assert dt(2000,'march', 1) == datetime.datetime(2000,3,1)
    >>> assert dt(2000,'h', 1) == datetime.datetime(2000,3,1) # future codes


    :Example: date as offset from today
    -----------------------------------
    >>> today = dt(0); 
    >>> import datetime
    >>> day = datetime.timedelta(1)
    >>> assert dt(-3) == today - 3 * day
    >>> assert dt('-10b') == today - 14 * day
    
    :Example: datetime arithmetic:
    -----------------------------------------------
    dt has an interesting logic in implementing datetime arithmentic: 
        
        - day and month parameters can be negative or bigger than the days of month
        - dt() will roll back/forward from the date which is valid
    
    >>> assert dt(2000,4,1) == datetime.datetime(2000, 4, 1, 0, 0)
    >>> assert dt(2000,4,0) == datetime.datetime(2000, 3, 31, 0, 0) # a day before dt(2000,4,1)

    and rolling back months:
        
    >>> assert dt(2000,0,1) == datetime.datetime(1999, 12, 1, 0, 0) # a month before dt(2000,1,1)
    >>> assert dt(2000,13,1) == datetime.datetime(2001, 1, 1, 0, 0) # a month after dt(2000,12,1)
    
    This may feel unnatural at first, but does allow for much nicer code, e.g.:
    [dt(2000,i,1) for i in range(-10,10)]

    :Parameters:
    ----------------
    *args : str, int or dates
        argument to be converted into dates
    dialect : str, optional
        parsing of 01/02/2020 is it 1st Feb or 2nd Jan? The default is 'uk', i.e. dd/mm/yyyy
    none : callable, optional
        What is dt()? The default is datetime.datetime.now()
    
    """
    args = as_list(args)
    if len(args) == 0:
        res = none() if callable(none) else none
        return tz_convert(res, tzinfo)
    t = args[0]
    if isinstance(t, np.datetime64):
        t = tz_convert(np2dt(t), tzinfo)
    elif isinstance(t, NaTType):
        return NaT
    elif isinstance(t, datetime.date) and not isinstance(t, datetime.datetime):
        t = datetime.datetime(t.year, t.month, t.day, tzinfo = as_tz(tzinfo))
    elif is_ts(t):
        return dt_bump(t, *args[1:])
    if isinstance(t, datetime.datetime):
        return reduce(dt_bump, args[1:], tz_convert(t, tzinfo))
    if len(args) == 1:
        if t is None:
            return tz_convert(none(), tzinfo) if callable(none) else none
        elif is_num(t):
            if is_nan(t):
                return tz_convert(none(), tzinfo) if callable(none) else none
            else:                
                return tz_convert(num2dt(t), tzinfo)
        elif is_bump(t):
            return dt_bump(dt(0, tzinfo = tzinfo), t)
        elif is_str(t):
            return uk2dt(t, tzinfo = tzinfo) if dialect == 'uk' else us2dt(t, tzinfo = tzinfo)
                # return int2dt(int(t)) + datetime.timedelta(float(t) % 1)
        else:
            raise ValueError('date format unrecognised %s'%t)
    elif len(args) == 2:
        y,m = ym(*args)
        return datetime.datetime(y,m,1, tzinfo = as_tz(tzinfo))
    y,m,d = args[:3]
    t = _ymd(y,m,d)
    if len(args) == 4 and is_str(args[3]):
        return tz_convert(nth_weekday_of_month(*args), tzinfo)
    if len(args) > 3:
        args = [int(a) for a in args[3:]] + [0,0,0]
        return tz_convert(t + datetime.timedelta(hours = args[0], minutes = args[1], seconds = args[2]), tzinfo)
    else:
        return tz_convert(t, tzinfo)


def nth_weekday_of_month(y, m, n, w):
    """
    y = 2020; m = 2
    assert nth_weekday_of_month(y, m, -1, 'sat') == dt(2020, 2, 29)
    assert nth_weekday_of_month(y, m, -2, 'sat') == dt(2020, 2, 22)
    assert nth_weekday_of_month(y, m, 1, 'sat') == dt(2020, 2, 1)
    assert nth_weekday_of_month(y, m, 1, 'sun') == dt(2020, 2, 2)
    assert nth_weekday_of_month(y, m, 1, 'monday') == dt(2020, 2, 3)
    assert nth_weekday_of_month(y, 'G', 3, 'sat') == dt(2020, 2, 15)
    assert nth_weekday_of_month(y, 'G', 3, 'sun') == dt(2020, 2, 16)
    assert nth_weekday_of_month(y, 'G', 3, 'monday') == dt(2020, 2, 17)
    """
    if n < 0 :
        return nth_weekday_of_month(y, m+1, 1, w) + datetime.timedelta(7 * n)
    t = dt(y, m , 1)
    bump = wkdays[w[:3].lower()] - t.weekday()
    if bump < 0:
        bump = bump + 7
    bump = bump + (n-1) * 7 
    res = t + datetime.timedelta(bump)
    return res

def ymd(*args, dialect = 'uk', none = datetime.datetime.now, tzinfo = None):
    """
    just like dt() but always returns date only (year/month/date) without fractions.
    see dt() for full documentation
    
    Returns
    -------
    datetime.datetime

    """
    t = dt(*args, dialect = dialect , none = none, tzinfo = tzinfo)
    return datetime.datetime(t.year, t.month, t.day, tzinfo = t.tzinfo)


ndt = partial(dt, none = None)
ndt.now = datetime.datetime.now
ndt.today = today
ndt.timedelta = datetime.timedelta

dt.now = datetime.datetime.now
dt.today = today
dt.timedelta = datetime.timedelta


def _date_format(fmt = None):
    """
    convenience function to allow quick format string

    :Parameters:
    ----------------
    fmt : str
        DESCRIPTION. The default is None.

    :Returns:
    -------
    fmt : str
        A valid format string for dates.
        
    :Examples:
    ---------------
    _date_format('Y/M')

    """
    if fmt is None: 
        fmt = ''
    if is_str(fmt) and len(fmt)<=1:
        fmt = fmt.join(['%Y', '%m', '%d'])
    if '%' not in fmt:
        fmt = ''.join(['%'+x if x in 'aAwdbBmyYHIpMSfzZjUWcXx' else x for x in fmt])
    return fmt

def dt2str(t, fmt = None):
    """
    converts a date into a string format. fmt supports the formats as specified by datetime().strftime() but also simplifies it
    
    :Parameters:
    -----------------
    t : date
    fmt : a format string
    
        if fmt is None:
            If the date has no intra-day component, by defauult will go to yyyymmdd format.
            If the date has intra-day component, will go to iso-format.
    
    :Examples:
    ---------------
    >>> assert dt2str(2000)  == '20000101'
    >>> assert dt2str(2000, 'Ymd')  == '20000101'
    >>> assert dt2str(2000, '-') == '2000-01-01'
    >>> assert dt2str(2000, '%B %Y') == 'January 2000'
    >>> assert dt2str(2000, 'B Y') == 'January 2000'
    >>> assert dt2str(2000, 'iso') == '2000-01-01T00:00:00'
    >>> t = datetime.datetime(2000,1,10,20,30,40,50)
    >>> assert dt2str(t) == '2000-01-10T20:30:40.000050'
    """
    t = t if isinstance(t, datetime.datetime) else dt(t)
    if isinstance(t, NaTType):
        return 'NaT'
    if fmt is None:
        if t == today(t):
            return t.strftime('%Y%m%d')
        else:
            return t.isoformat()
    fmt = _date_format(fmt)
    if fmt.lower() == 'iso':
        return t.isoformat()
    else:
        return t.strftime(fmt)

    
