import os
import sys

_default_backend = 'numpy'
if 'GEOMSTATS_BACKEND' in os.environ:
    _backend = os.environ['GEOMSTATS_BACKEND']

else:
    _backend = _default_backend

_BACKEND = _backend

if _BACKEND == 'numpy':
    sys.stderr.write('Using numpy backend\n')
    from .numpy_backend import *  # NOQA
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using tensorflow backend\n')
    from .tensorflow_backend import *  # NOQA
elif _BACKEND == 'pytorch':
    raise NotImplementedError('pytorch backend not implemented yet')


def backend():
    return _BACKEND
