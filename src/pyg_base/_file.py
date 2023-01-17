import os
from pyg_npy import mkdir, path_name, path_dirname, path_join
import csv

__all__ = ['path_name', 'path_dirname', 'path_join', 'mkdir', 'read_csv']

    
def read_csv(path, encoding = None, errors = 'replace', **fmt):
    """
    A light-weight csv reader, no conversion is done, nor do we insist equal number of columns per row.
    - by default, encoding error (unicode characters) are replaced.
    - fmt parameters are parameters for the csv.reader object, see https://docs.python.org/3/library/csv.html
    
    
    Parameters:
    -----------
    path: str/list of str
        file(s) to read or list of a directory
    
    errors: 
        how to handle errors when opening a file. See io.open for full help

    encoding:
        file encoding. See io.open for full help
    
    **fmt: 
        formatting options fed into csv.reader, for example dialect = 'excel'
    
    """        
    if isinstance(path, list):
        return [read_csv(p, encoding = encoding, errors = errors, **fmt) for p in path]
    elif isinstance(path, dict):
        return {k: read_csv(p, encoding = encoding, errors = errors, **fmt) for k,p in path.items()}
    path = path_name(path)
    if not os.path.exists(path) and not path.endswith('.csv') and os.path.exists(path + '.csv'):
        path = path + '.csv'
    with open(path, 'r', errors = errors, encoding = encoding) as f:
        reader = csv.reader(f, **fmt)
        data = list(reader)
    return data


def _listdir(path):
    try: 
        return os.listdir(path)
    except PermissionError:
        return []


def dictdir(path, level = 0, skip_permission = True):
    """
    returns a tree like structure where the leaf of each node is the path. 

    Parameters:
    -----------
    path: str
        initial path to os.listdir
    
    level: int
        how many levels for the subdir to explore

    skip_permission: bool
        if True, will skip directories with permission errors, treating them as empty directories
        
    Returns:
    --------
    tree
    
    Example
    -------
    >>> path = 'c:/'
    >>> assert list(dictdir(path,0)) == os.listdir(path)
    
    >>> dictdir(path)

    {'conda': 'c:/conda',
     'data': 'c:/data',
     'Dell': 'c:/Dell',
     'Documents and Settings': 'c:/Documents and Settings',
     'github': 'c:/github',
     'Program Files': 'c:/Program Files',
     'Program Files (x86)': 'c:/Program Files (x86)',
     'ProgramData': 'c:/ProgramData',
     'Users': 'c:/Users',
     'Windows': 'c:/Windows',}


    Example:
    --------
    I keep my barchart trading data here, for each ticker there are multiple futures expiring at different year & month

    >>> path = 'C:/Users/Dell/Dropbox/Yoav/TradingData/bc'
    >>> res = dictdir(path,3)


    We can now easily query the data: Here is where I keep all the intraday files:

    >>> dictable(res, '%ticker/%y/%m/intraday/%path')[['ticker','y','m','path']]
    
    dictable[6194 x 4]
    ticker     |y   |m|path                                              
    1UA Index |2023|F|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\1UA Inde
    1UA Index |2023|G|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\1UA Inde
    2UA Index |2023|F|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\2UA Inde
    ...3823 rows...
    ZWPA Index|2025|M|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\ZWPA Ind
    ZWPA Index|2025|U|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\ZWPA Ind
    ZWPA Index|2025|Z|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\ZWPA Ind


    ... and here is where I keep all the end-of-day files:
        
    >>> dictable(res, '%ticker/%y/%m/eod/%path')[['ticker','y','m','path']]

    dictable[6182 x 4]
    ticker     |y   |m|path                                              
    1UA Index  |2023|F|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\1UA Inde
    1UA Index  |2023|G|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\1UA Inde
    2UA Index  |2023|F|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\2UA Inde
    ...6182 rows...
    ZZPA Comdty|2023|V|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\ZZPA Com
    ZZPA Comdty|2023|X|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\ZZPA Com
    ZZPA Comdty|2023|Z|C:/Users/Dell/Dropbox/Yoav/TradingData/bc\ZZPA Com

    """
    path = path_name(path)
    res = {x : os.path.join(path, x) for x in (_listdir if skip_permission else os.listdir)(path)}
    if level>0:
        res = {k : dictdir(v, level = level-1, skip_permission = skip_permission) if os.path.isdir(v) else v for k, v in res.items()}
    return res





