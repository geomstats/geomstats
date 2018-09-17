__version__ = '1.11-dev'  # Remove -dev before releasing

install_requires = [
    'autograd',
    'numpy>=1.14.1',
    'scipy',
    'matplotlib',
    ]

extras_require = {
    'test': ['nose2', 'coverage', 'codecov'],
    'tf': ['tensorflow>=1.8'],
    'torch': ['torch==0.4.0'],
    }
extras_require['all'] = sum(extras_require.values(), [])
