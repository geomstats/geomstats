import os
import sys

from numpy import pi  # NOQA


_BACKEND = os.environ.get('GEOMSTATS_BACKEND')
if _BACKEND is None:
    os.environ['GEOMSTATS_BACKEND'] = _BACKEND = 'numpy'

if _BACKEND == 'numpy':
    from geomstats.backend.numpy import *  # NOQA
elif _BACKEND == 'pytorch':
    from geomstats.backend.pytorch import *  # NOQA
elif _BACKEND == 'tensorflow':
    from geomstats.backend.tensorflow import *  # NOQA
else:
    raise RuntimeError('Unknown backend \'{:s}\''.format(_BACKEND))
print('Using {:s} backend'.format(_BACKEND), file=sys.stderr)
