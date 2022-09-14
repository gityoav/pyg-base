import os
import json
CFG = os.environ.get('PYG_CFG')

def mkdir(path):
    """
    sames as os.mkdir but 
    1) allows for subdir i.e.:
    2) will not attempt to create if path exits, hence can run safely


    :Example:
    ---------
    >>> with pytest.raises(Exception):       
    >>>     os.mkdir('c:/temp/some/subdir/that/we/want')

    >>> print('but', mkdir('c:/temp/some/subdir/that/we/want'), 'now exists')
    
    """
    if not os.path.isdir(path):
        p = path.replace('\\', '/')
        paths = p.split('/')
        if '.' in paths[-1]:
            paths = paths[:-1]
        for i in range(2, len(paths)+1):
            d = '/'.join(paths[:i])
            if not os.path.isdir(d):
                os.mkdir(d)
    return path

CACHE = {}

def get_cache(*args):
    res = CACHE
    for arg in args:
        if arg not in res:
            res[arg] = {}
        res = res[arg]
    return res

def cfg_read():
    cfg = CACHE.get('CFG', {})
    if isinstance(CFG, str):
        for path in CFG.split(','):
            if os.path.isfile(path):
                with open(path, 'r') as f:
                    cfg.update(json.load(f))
                    CACHE['CFG'] = cfg
    return cfg

cfg_read.__doc__ = f'reads the config from get_cache("CFG") and updates it from files in {CFG}'


def cfg_write(cfg):
    CACHE['CFG'] = cfg
    if CFG is not None:
        for path in CFG.split(','):
            try:
                with open(mkdir(path), 'w') as f:
                    json.dump(cfg, f)
                return
            except Exception:
                pass
   
cfg_write.__doc__ = f'writes the config file provided to {CFG}' 
    