import functools

import torch as _torch
from torch import complex64, complex128, float32, float64

from geomstats._backend import _backend_config as _config
from geomstats._backend._dtype_utils import (
    _MAP_FLOAT_TO_COMPLEX,
    _pre_add_default_dtype,
    get_default_dtype,
)

from ._common import cast

MAP_DTYPE = {
    "float32": float32,
    "float64": float64,
    "complex64": complex64,
    "complex128": complex128,
}


def as_dtype(value):
    return MAP_DTYPE[value]


def set_default_dtype(value):
    _config._DEFAULT_DTYPE = as_dtype(value)
    _config._DEFAULT_COMPLEX_DTYPE = _MAP_FLOAT_TO_COMPLEX.get(value)
    _torch.set_default_dtype(_config._DEFAULT_DTYPE)

    return _config._DEFAULT_DTYPE


_add_default_dtype = _pre_add_default_dtype(cast)


def _preserve_input_dtype(target=None):
    # only acts on input
    # assumes dtype is kwarg
    # use together with _add_default_dtype

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, dtype=None, **kwargs):
            if dtype is None:
                dtype = x.dtype

            return func(x, *args, dtype=dtype, **kwargs)

        return _wrapped

    if target is None:
        return _decorator
    else:
        return _decorator(target)


def _box_unary_scalar(target=None):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):
            if not _torch.is_tensor(x):
                x = _torch.tensor(x)
            return func(x, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator
    else:
        return _decorator(target)


def _box_binary_scalar(target=None, box_x1=True, box_x2=True):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x1, x2, *args, **kwargs):
            if box_x1 and not _torch.is_tensor(x1):
                x1 = _torch.tensor(x1)
            if box_x2 and not _torch.is_tensor(x2):
                x2 = _torch.tensor(x2)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator
    else:
        return _decorator(target)
