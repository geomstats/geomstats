import logging
import os

from numpy import pi  # NOQA

logging.basicConfig(format='%(levelname)s: %(message)s')
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

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
if loggers and loggers[0].name.startswith('nose2'):
    logging.getLogger().setLevel(logging.WARNING)
