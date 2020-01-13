import os
import sys

from numpy import pi


_BACKEND = os.environ.get('GEOMSTATS_BACKEND')
if _BACKEND is None:
    os.environ['GEOMSTATS_BACKEND'] = _BACKEND = 'numpy'


if _BACKEND == 'numpy':
    sys.stderr.write('Using numpy backend\n')
    from .numpy import *  # NOQA
    from . import numpy_linalg as linalg
    from . import numpy_random as random
    from . import numpy_testing as testing
elif _BACKEND == 'pytorch':
    sys.stderr.write('Using pytorch backend\n')
    from .pytorch import *  # NOQA
    from . import pytorch_linalg as linalg  # NOQA
    from . import pytorch_random as random  # NOQA
    from . import pytorch_testing as testing  # NOQA
elif _BACKEND == 'tensorflow':
    sys.stderr.write('Using tensorflow backend\n')
    from .tensorflow import *  # NOQA
    from . import tensorflow_linalg as linalg  # NOQA
    from . import tensorflow_random as random  # NOQA
    from . import tensorflow_testing as testing  # NOQA
else:
    sys.stderr.write('Unknown backend \'{:s}\'\n'.format(_BACKEND))
    sys.exit(1)
