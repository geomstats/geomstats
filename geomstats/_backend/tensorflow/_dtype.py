import functools

import tensorflow as _tf
from tensorflow.dtypes import as_dtype

from geomstats._backend import _backend_config as _config
from geomstats._backend._dtype_utils import (
    _dyn_update_dtype,
    _modify_func_default_dtype,
    _pre_allow_complex_dtype,
    _pre_cast_out_from_dtype,
    _pre_cast_out_to_input_dtype,
    _pre_set_default_dtype,
    _update_default_dtypes,
    get_default_cdtype,
    get_default_dtype,
)

_COMPLEX_DTYPES = (
    _tf.complex64,
    _tf.complex128,
)


def is_floating(x):
    return x.dtype.is_floating


def is_complex(x):
    return x.dtype.is_complex


def is_bool(x):
    return x.dtype.is_bool


def _dtype_as_str(dtype):
    return dtype.name


set_default_dtype = _pre_set_default_dtype(as_dtype)

_cast_out_to_input_dtype = _pre_cast_out_to_input_dtype(
    _tf.cast, is_floating, is_complex, as_dtype, _dtype_as_str
)

_cast_out_from_dtype = _pre_cast_out_from_dtype(_tf.cast, is_floating, is_complex)


def _allow_complex_dtype(target=None):
    """Allow complex type by calling the function twice.

    Assumes function do not support dtype.

    How it works?
    -------------
    Function is called twice if dtype is complex.
    Output is casted if not corresponding to expected dtype.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            dtype = kwargs.get("dtype")

            if dtype not in _COMPLEX_DTYPES:
                return func(*args, **kwargs)

            del kwargs["dtype"]
            real = _tf.cast(func(*args, **kwargs), dtype)
            imag = 1j * _tf.cast(func(*args, **kwargs), dtype)

            return real + imag

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _box_unary_scalar(target=None):
    """Update dtype if input is float for unary operations.

    How it works?
    -------------
    Promotes input to tensorfow constant if not the case.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x, *args, **kwargs):

            if type(x) is float:
                x = _tf.constant(x, dtype=_config.DEFAULT_DTYPE)
            elif type(x) is complex:
                x = _tf.constant(x, dtype=_config.DEFAULT_COMPLEX_DTYPE)

            return func(x, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)


def _box_binary_scalar(target=None):
    """Update dtype if input is float for binary operations.

    How it works?
    -------------
    Promotes input to tensorfow constant if not the case.
    """

    def _decorator(func):
        @functools.wraps(func)
        def _wrapped(x1, x2, *args, **kwargs):

            if type(x1) is float:
                x1 = _tf.constant(x1, dtype=_config.DEFAULT_DTYPE)

            if type(x2) is float:
                x2 = _tf.constant(x2, dtype=_config.DEFAULT_DTYPE)

            return func(x1, x2, *args, **kwargs)

        return _wrapped

    if target is None:
        return _decorator

    return _decorator(target)
