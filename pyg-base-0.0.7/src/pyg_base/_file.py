import os
import shutil
import re
from pyg_base._logger import logger
from pyg_npy import mkdir, path_name, path_dirname, path_join
import subprocess
import csv

__all__ = ['path_name', 'path_dirname', 'path_join', 'mkdir', 'read_csv']

    
def read_csv(path, errors = 'replace', **fmt):
    """
    A light-weight csv reader, no conversion is done, nor do we insist equal number of columns per row.
    - by default, encoding error (unicode characters) are replaced.
    - fmt parameters are parameters for the csv.reader object, see https://docs.python.org/3/library/csv.html
    """
    path = path_name(path)
    with open(path, 'r', errors = errors) as f:
        reader = csv.reader(f, **fmt)
        data = list(reader)
    return data