from copy import copy
from pyg_base._dictable import dictable
from pyg_base._as_list import as_list
from pyg_base._dict import _tree_setitem, dictattr, _tree_types


def _table_to_tree(tree, pattern, d, base, ignore = None, types = None):
    path = pattern.split('/')
    item = [d[p[1:]] if p.startswith('%') else p for p in path]
    _tree_setitem(tree, item, base = base, ignore = ignore, types = types)
    

def table_to_tree(tree, pattern, table, base = dictattr, ignore = None, types = None):
    """
    This is the reverse of tree_to_table and allows us to modify a tree using a table of values.
    
    We set an item that looks like this:

    >>> pattern = 'markets/%market/weight/%weight' 
    
    with a value that looks like this:
        
    >>> table = dictable(market = ['TY','ES'], weight = [0.3,0.7])
    
    >>> table_to_tree(None, pattern, table = table)
    
    {'markets': {'TY': {'weight': 0.3}, 'ES': {'weight': 0.7}}}
    
    
    """
    types = _tree_types(types)
    ignore = as_list(ignore)
    base = type(tree) if base is None else base
    tree = base() if tree is None else copy(tree)
    if isinstance(table, (dictable, list)):
        for row in table:
            _table_to_tree(tree, pattern, row, base = base, ignore = ignore, types = types)
    else:
        _table_to_tree(tree, pattern, table, base = base, ignore = ignore, types = types)
    return tree
    


