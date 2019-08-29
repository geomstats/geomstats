# Remove -dev before releasing
__version__ = '1.16'

from itertools import chain

install_requires = [
    'autograd',
    'h5py==2.8.0',
    'matplotlib',
    'numpy>=1.14.1',
    'scipy',
    ]

extras_require = {
    'test': ['codecov', 'coverage', 'nose2'],
    'tf': ['tensorflow>=1.12'],
    'torch': ['torch==0.4.0'],
    }
extras_require['all'] = list(chain(*extras_require.values()))
