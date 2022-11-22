import functools

import torch as _torch
from torch import complex64, complex128, float32, float64

from geomstats._backend import _backend_config as _config
from geomstats._backend._dtype_utils import (
    _MAP_FLOAT_TO_COMPLEX,
    _modify_func_default_dtype,
    _pre_add_default_dtype_by_casting,
    _pre_allow_complex_dtype,
    _pre_cast_out_to_input_dtype,
    _update_default_dtypes,
    get_default_cdtype,
    get_default_dtype,
)

from ._common import cast

MAP_DTYPE = {
    "float32": float32,
    "float64": float64,
    "complex64": complex64,
    "complex128": complex128,
}

_COMPLEX_DTYPES = (complex64, complex128)


def is_floating(x):
    return x.dtype.is_floating_point


def is_complex(x):
    return x.dtype.is_complex


def is_bool(x):
    return x.dtype is _torch.bool


def as_dtype(value):
    """Transform string representing dtype in dtype."""
    return MAP_DTYPE[value]


def _dtype_as_str(dtype):
    return str(dtype).split(".")[-1]


def set_default_dtype(value):
    """Set backend default dtype.

    Parameters
    ----------
    value : str
        Possible values are "float32" as "float64".
    """
    _config.DEFAULT_DTYPE = as_dtype(value)
    _config.DEFAULT_COMPLEX_DTYPE = as_dtype(_MAP_FLOAT_TO_COMPLEX.get(value))
    _torch.set_default_dtype(_config.DEFAULT_DTYPE)

    _update_default_dtypes()

    return _config.DEFAULT_DTYPE


_add_default_dtype_by_casting = _pre_add_default_dtype_by_casting(cast)
_cast_out_to_input_dtype = _pre_cast_out_to_input_dtype(
    cast, is_floating, is_complex, as_dtype, _dtype_as_str
)
_allow_complex_dtype = _pre_allow_complex_dtype(cast, _COMPLEX_DTYPES)


def _preserve_input_dtype(target=None):
    """Ensure input dtype is preserved.

    How it works?
    -------------
    Only acts on input. Assumes dtype is kwarg and function accepts dtype.
    Passes dtype as input dtype.

    Use together with `_add_default_dtype_by_casting`.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, dtype=None, **kwargs):
            if dtype is None:
                dtype = x.dtype

            return func(x, *args, dtype=dtype, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _box_unary_scalar(target=None):
    """Update dtype if input is float in unary operations.

    How it works?
    -------------
    Promotes input to tensor if not the case.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):
            if not _torch.is_tensor(x):
                x = _torch.tensor(x)
            return func(x, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _box_binary_scalar(target=None, box_x1=True, box_x2=True):
    """Update dtype if input is float in binary operations.

    How it works?
    -------------
    Promotes inputs to tensor if not the case.
    """

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

    return _decorator(target)
