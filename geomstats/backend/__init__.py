import os
import sys

from numpy import pi  # NOQA


_BACKEND = os.environ.get('GEOMSTATS_BACKEND')
if _BACKEND is None:
    os.environ['GEOMSTATS_BACKEND'] = _BACKEND = 'numpy'

if _BACKEND == 'numpy':
    print('Using numpy backend', file=sys.stderr)
    from geomstats.backend.numpy import *  # NOQA
elif _BACKEND == 'pytorch':
<<<<<<< HEAD
    print('Using pytorch backend', file=sys.stderr)
    from geomstats.backend.pytorch import *  # NOQA
elif _BACKEND == 'tensorflow':
    print('Using tensorflow backend', file=sys.stderr)
    from geomstats.backend.tensorflow import *  # NOQA
else:
    raise RuntimeError('Unknown backend \'{:s}\''.format(_BACKEND))
