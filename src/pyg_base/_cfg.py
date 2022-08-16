import os
import json
CFG = os.environ.get('PYG_CFG', 'e:/etc/pyg.json')
_backup = 'c:/etc/pyg.json'

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

def cfg_read():
    for path in [CFG, _backup]:
        if os.path.isfile(path):
            with open(path, 'r') as f:
                cfg = json.load(f)
            return cfg
    else:
        return {}
cfg_read.__doc__ = 'reads the config file from %s' % CFG



def cfg_write(cfg):
    try:
        with open(mkdir(CFG), 'w') as f:
            json.dump(cfg, f)
    except Exception:
        with open(mkdir(_backup), 'w') as f:
            json.dump(cfg, f)
        
cfg_write.__doc__ = 'writes the config file provided to %s' % CFG
    

