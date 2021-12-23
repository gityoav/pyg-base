from pyg_base._file import mkdir, path_name
from pyg_base._types import is_series, is_df, is_int, is_date, is_bool, is_str, is_float
from pyg_base._dates import dt2str, dt
from pyg_base._logger import logger
from pyg_base._as_list import as_list
from npy_append_array import NpyAppendArray as npa
import pandas as pd
import numpy as np
import jsonpickle as jp

_series = '_is_series'
_npy = '.npy'

__all__ = ['pd_to_parquet', 'pd_read_parquet', 'pd_to_npy', 'pd_read_npy', 'np_save']


def np_save(path, value, append = False):
    mkdir(path)
    if append:
        with npa(path) as f:
            f.append(value)
    else:
        np.save(path, value)
    return path


def pd_to_npy(value, path, append = False):
    """
    writes a pandas DataFrame/series to a collection of numpy files as columns.
    Support append rather than overwrite
    
    :Params:
    --------
    value: pd.DataFrame/Series
        value to be saved
    
    path: str
        location of the form c:/test/file.npy

    append: bool
        if True, will append to existing files rather than overwrite


    :Returns:
    ---------
    dict of path, columns and index
    These are the inputs needed by pd_read_npy
    
    :Example:
    ----------
    >>> import numpy as np   
    >>> import pandas as pd
    >>> from pyg_base import *
    >>> path = 'c:/temp/test.npy'
    >>> value = pd.DataFrame([[1,2],[3,4]], drange(-1), ['a', 'b'])
    >>> res = pd_to_npy(value, path)

    >>> res
    >>> {'path': 'c:/temp/test.npy', 'columns': ['a', 'b'], 'index': ['index']}

    >>> df = pd_read_npy(**res)    
    >>> assert eq(df, value)
    
    """
    res = dict(path = path)
    if is_series(value):
        df = pd.DataFrame(value)
        columns = list(df.columns)
        res['columns'] = columns[0]
    else:
        df = value
        res['columns'] = columns = list(df.columns)

    df = df.reset_index()
    res['index'] = list(df.columns)[:-len(columns)]    
    if path.endswith(_npy):
        path = path[:-len(_npy)]
    
    for col in df.columns:
        a = df[col].values
        fname = path +'/%s%s'%(col, _npy)
        np_save(fname, a, append)
    return res

pd_to_npy.output = ['path', 'columns', 'index']

def pd_read_npy(path, columns, index):
    """
    reads a pandas dataframe/series from a path directory containing npy files with col.npy and idx.npy names

    Parameters
    ----------
    path : str
        directory where files are.
    columns : str/list of str
        filenames for columns. If columns is a single str, assumes we want a pd.Series
    index : str/list of str
        column names used as indices

    Returns
    -------
    res : pd.DataFrame/pd.Series
    
    """
    if path.endswith(_npy):
        path = path[:-len(_npy)]
    data = {col : np.load(path +'/%s%s'%(col, _npy)) for col in as_list(columns) + as_list(index)}
    res = pd.DataFrame(data).set_index(index)
    if isinstance(columns, str): # it is a series
        res = res[columns]
    return res

def pd_to_parquet(value, path, compression = 'GZIP'):
    """
    a small utility to save df to parquet, extending both pd.Series and non-string columns    

    :Example:
    -------
    >>> from pyg_base import *
    >>> import pandas as pd
    >>> import pytest

    >>> df = pd.DataFrame([[1,2],[3,4]], drange(-1), columns = [0, dt(0)])
    >>> s = pd.Series([1,2,3], drange(-2))

    >>> with pytest.raises(ValueError): ## must have string column names
            df.to_parquet('c:/temp/test.parquet')

    >>> with pytest.raises(AttributeError): ## pd.Series has no to_parquet
            s.to_parquet('c:/temp/test.parquet')
    
    >>> df_path = pd_to_parquet(df, 'c:/temp/df.parquet')
    >>> series_path = pd_to_parquet(s, 'c:/temp/series.parquet')

    >>> df2 = pd_read_parquet(df_path)
    >>> s2 = pd_read_parquet(series_path)

    >>> assert eq(df, df2)
    >>> assert eq(s, s2)

    """
    if is_series(value):
        mkdir(path)
        df = pd.DataFrame(value)
        df.columns = [_series]
        try:
            df.to_parquet(path, compression = compression)
        except ValueError:
            df = pd.DataFrame({jp.dumps(k) : [v] for k,v in dict(value).items()})
            df[_series] = True
            df.to_parquet(path, compression = compression)
        return path
    elif is_df(value):
        mkdir(path)
        df = value.copy()
        df.columns = [jp.dumps(col) for col in df.columns]
        df.to_parquet(path, compression  = compression)
        return path
    else:
        return value        

def pd_read_parquet(path):
    """
    a small utility to read df/series from parquet, extending both pd.Series and non-string columns 

    :Example:
    -------
    >>> from pyg import *
    >>> import pandas as pd
    >>> import pytest

    >>> df = pd.DataFrame([[1,2],[3,4]], drange(-1), columns = [0, dt(0)])
    >>> s = pd.Series([1,2,3], drange(-2))

    >>> with pytest.raises(ValueError): ## must have string column names
            df.to_parquet('c:/temp/test.parquet')

    >>> with pytest.raises(AttributeError): ## pd.Series has no to_parquet
            s.to_parquet('c:/temp/test.parquet')
    
    >>> df_path = pd_to_parquet(df, 'c:/temp/df.parquet')
    >>> series_path = pd_to_parquet(s, 'c:/temp/series.parquet')

    >>> df2 = pd_read_parquet(df_path)
    >>> s2 = pd_read_parquet(series_path)

    >>> assert eq(df, df2)
    >>> assert eq(s, s2)

    """
    path = path_name(path)
    try:
        df = pd.read_parquet(path)
    except Exception:
        logger.warning('WARN: unable to read pd.read_parquet("%s")'%path)
        return None
    if df.columns[-1] == _series:
        if len(df.columns) == 1:
            res = df[_series]
            res.name = None
            return res
        else:
            return pd.Series({jp.loads(k) : df[k].values[0] for k in df.columns[:-1]})
    else:
        df.columns = [jp.loads(col) for col in df.columns]
        return df

