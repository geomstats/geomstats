import logging
import os
import sys

BACKEND_FUNCTIONS = {
    '': [
        # Types
        'int32',
        'int64',
        'float32',
        'float64',
        # Functions
        'abs',
        'all',
        'allclose',
        'amax',
        'amin',
        'arange',
        'arccos',
        'arccosh',
        'arcsin',
        'arctan2',
        'arctanh',
        'argmax',
        'argmin',
        'array',
        'asarray',
        'cast',
        'ceil',
        'clip',
        'concatenate',
        'cond',
        'copy',
        'cos',
        'cosh',
        'cumsum',
        'diag',
        'diagonal',
        'divide',
        'dot',
        'einsum',
        'empty',
        'equal',
        'eval',
        'exp',
        'expand_dims',
        'eye',
        'flip',
        'floor',
        'gather',
        'get_mask_i_float',
        'greater',
        'hsplit',
        'hstack',
        'isclose',
        'less',
        'less_equal',
        'linspace',
        'log',
        'logical_or',
        'matmul',
        'maximum',
        'mean',
        'meshgrid',
        'mod',
        'ndim',
        'ones',
        'ones_like',
        'outer',
        'repeat',
        'reshape',
        'shape',
        'sign',
        'sin',
        'sinh',
        'split',
        'sqrt',
        'squeeze',
        'stack',
        'std',
        'sum',
        'tan',
        'tanh',
        'tile',
        'to_ndarray',
        'trace',
        'transpose',
        'triu_indices',
        'vectorize',
        'vstack',
        'where',
        'while_loop',
        'zeros',
        'zeros_like'
    ],
    'linalg': [
        'det',
        'eig',
        'eigh',
        'eigvalsh',
        'expm',
        'inv',
        'logm',
        'norm',
        'powerm',
        'qr',
        'sqrtm',
        'svd'
    ],
    'random': [
        'choice',
        'normal',
        # TODO(nkoep): Remove 'rand' and replace it by 'uniform'. Much like
        #              'randn' is a convenience wrapper (which we don't use)
        #              around 'normal', 'rand' only wraps 'uniform'.
        'rand',
        'randint',
        'seed',
        'uniform'
    ]
}


class BackendImporter:
    """Importer class to create the backend module."""

    def __init__(self, path):
        self._path = path

    def _verify_backend_module(self, backend_name):
        if backend_name == 'numpy':
            from geomstats._backend import numpy as backend
        elif backend_name == 'pytorch':
            from geomstats._backend import pytorch as backend
        elif backend_name == 'tensorflow':
            from geomstats._backend import tensorflow as backend
        else:
            raise RuntimeError('Unknown backend \'{:s}\''.format(backend_name))

        for module_name, attributes in BACKEND_FUNCTIONS.items():
            if module_name:
                try:
                    module = getattr(backend, module_name)
                except AttributeError:
                    raise RuntimeError(
                        'Backend \'{}\' exposes no \'{}\' submodule'.format(
                            backend_name, module_name)) from None
            else:
                module = backend
            for attribute_name in attributes:
                try:
                    getattr(module, attribute_name)
                except AttributeError:
                    if module_name:
                        error = (
                            'Submodule \'{}\' of backend \'{}\' provides no '
                            'attribute \'{}\''.format(
                                module_name, backend_name, attribute_name))
                    else:
                        error = (
                            'Backend \'{}\' provides no attribute '
                            '\'{}\''.format(backend_name, attribute_name))
                    raise RuntimeError(error) from None

        from numpy import pi
        backend.pi = pi

        return backend

    def find_module(self, fullname, path=None):
        if self._path != fullname:
            return None
        return self

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]

        _BACKEND = os.environ.get('GEOMSTATS_BACKEND')
        if _BACKEND is None:
            os.environ['GEOMSTATS_BACKEND'] = _BACKEND = 'numpy'

        module = self._verify_backend_module(_BACKEND)
        module.__loader__ = self
        sys.modules[fullname] = module

        logging.info('Using {:s} backend'.format(_BACKEND))
        return module

sys.meta_path.append(BackendImporter('geomstats.backend'))
