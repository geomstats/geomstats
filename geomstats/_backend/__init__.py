import logging
import os
import sys
import types


class BackendImporter:
    """Importer class to create the backend module."""

    def __init__(self, path):
        self._path = path

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

        if _BACKEND == 'numpy':
            from geomstats._backend import numpy as backend
        elif _BACKEND == 'pytorch':
            from geomstats._backend import pytorch as backend
        elif _BACKEND == 'tensorflow':
            from geomstats._backend import tensorflow as backend
        else:
            raise RuntimeError('Unknown backend \'{:s}\''.format(_BACKEND))

        module = backend
        from numpy import pi
        module.pi = pi

        module.__file__ = '<{}>'.format(fullname)
        module.__loader__ = self
        sys.modules[fullname] = module

        logging.info('Using {:s} backend'.format(_BACKEND))
        return module

sys.meta_path.append(BackendImporter('geomstats.backend'))
