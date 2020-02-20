import logging
import os

from numpy import pi  # NOQA

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


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
logging.info('Using {:s} backend'.format(_BACKEND))
