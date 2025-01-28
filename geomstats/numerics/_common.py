import functools
import inspect
from types import MethodType

import numpy as np
from scipy.interpolate import PPoly

import geomstats.backend as gs


def result_to_backend_type(result):
    """Convert np.array to gs.array within result object."""
    if gs.__name__.endswith("numpy") or gs.__name__.endswith("autograd"):
        return result

    for key, value in result.items():
        if type(value) is np.ndarray:
            result[key] = gs.from_numpy(value)

        if isinstance(value, PPoly):
            new_ppoly = _InstanceConvertOutputWrapper(value)
            result[key] = new_ppoly

    return result


def _convert_np_output(func):
    @functools.wraps(func)
    def _wrapped(*args, **kwargs):
        out = func(*args, **kwargs)

        if type(out) is np.ndarray:
            return gs.from_numpy(out)

        return out

    return _wrapped


class _InstanceConvertOutputWrapper:
    """Dynamic wrapper for an instance to convert method output to gs.array."""

    def __init__(self, instance):
        self._instance = instance
        self._dict = {}

    def __getattr__(self, attr_name):
        if attr_name in self._dict:
            return self._dict[attr_name]

        attr = getattr(self._instance, attr_name)
        if isinstance(attr, MethodType):
            attr = _convert_np_output(attr)
            self._dict[attr_name] = attr

        return attr

    def __call__(self, *args, **kwargs):
        out = self._instance(*args, **kwargs)
        if type(out) is np.ndarray:
            return gs.from_numpy(out)

        return out

    def __dir__(self):
        return dir(self._instance)

    def __repr__(self):
        return repr(self._instance)

    def __str__(self):
        return str(self._instance)


def params_to_kwargs(obj, ignore=(), renamings=None, ignore_private=False, func=None):
    """Get dict with selected object attributes.

    Parameters
    ----------
    obj : object
        Object with desired attributes.
    ignore : tuple[str]
        Attributes to ignore.
    renamings: dict
        Attribute renamings.
    ignore_private: bool
        Whether to ignore private attributes.
    func : callable
        Function to get signature from. Attributes
        not in the signature are ignored.

    Returns
    -------
    kwargs : dict
    """
    kwargs = obj.__dict__.copy()

    if func is not None:
        params = inspect.signature(func).parameters
        ignore = list(ignore) + [key for key in kwargs if key not in params]

    if ignore:
        for key in ignore:
            kwargs.pop(key)

    if renamings is not None:
        for old_key, new_key in renamings.items():
            kwargs[new_key] = kwargs.pop(old_key)

    if ignore_private:
        private_keys = list(filter(lambda key: key.startswith("_"), kwargs.keys()))
        for key in private_keys:
            kwargs.pop(key)

    return kwargs
