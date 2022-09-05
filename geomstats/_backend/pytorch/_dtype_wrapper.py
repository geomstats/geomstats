import functools

import torch as _torch
from torch import float32, float64

from ._common import cast

_DEFAULT_DTYPE = None

MAP_DTYPE = {
    "float32": float32,
    "float64": float64,
}


def as_dtype(value):
    return MAP_DTYPE[value]


def set_default_dtype(value):
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = as_dtype(value)
    _torch.set_default_dtype(_DEFAULT_DTYPE)

    return get_default_dtype()


def get_default_dtype():
    return _DEFAULT_DTYPE


def _add_default_dtype(func):
    @functools.wraps(func)
    def _wrapped(*args, dtype=None, **kwargs):
        if dtype is None:
            dtype = _DEFAULT_DTYPE

        out = func(*args, **kwargs)
        if out.dtype != dtype:
            return cast(out, dtype)
        return out

    return _wrapped


def _preserve_input_dtype(func):
    # only acts on input
    # assumes dtype is kwarg
    # use together with _add_default_dtype

    @functools.wraps(func)
    def _wrapped(x, *args, dtype=None, **kwargs):
        if dtype is None:
            dtype = x.dtype

        return func(x, *args, dtype=dtype, **kwargs)

    return _wrapped
