from geomstats._backend._dtype_utils import _np_box_binary_scalar as _box_binary_scalar
from geomstats._backend._dtype_utils import _np_box_unary_scalar as _box_unary_scalar
from geomstats._backend._dtype_utils import (
    _pre_add_default_dtype_by_casting,
    _pre_allow_complex_dtype,
    _pre_cast_fout_to_input_dtype,
    _pre_cast_out_from_dtype,
    _pre_cast_out_to_input_dtype,
    _pre_set_default_dtype,
)

from .._backend_config import np_atol as atol
from .._backend_config import np_rtol as rtol
from ._dispatch import numpy as _np

_DTYPES = {
    _np.dtype("int32"): 0,
    _np.dtype("int64"): 1,
    _np.dtype("float32"): 2,
    _np.dtype("float64"): 3,
    _np.dtype("complex64"): 4,
    _np.dtype("complex128"): 5,
}

_COMPLEX_DTYPES = [
    _np.complex64,
    _np.complex128,
]


def is_floating(x):
    return x.dtype.kind == "f"


def is_complex(x):
    return x.dtype.kind == "c"


def is_bool(x):
    return x.dtype.kind == "b"


def as_dtype(value):
    """Transform string representing dtype in dtype."""
    return _np.dtype(value)


def _dtype_as_str(dtype):
    return dtype.name


def cast(x, dtype):
    return x.astype(dtype)


set_default_dtype = _pre_set_default_dtype(as_dtype)

_add_default_dtype_by_casting = _pre_add_default_dtype_by_casting(cast)
_cast_fout_to_input_dtype = _pre_cast_fout_to_input_dtype(cast, is_floating)
_cast_out_to_input_dtype = _pre_cast_out_to_input_dtype(
    cast, is_floating, is_complex, as_dtype, _dtype_as_str
)


_cast_out_from_dtype = _pre_cast_out_from_dtype(cast, is_floating, is_complex)
_allow_complex_dtype = _pre_allow_complex_dtype(cast, _COMPLEX_DTYPES)


def is_array(x):
    return type(x) is _np.ndarray


def to_ndarray(x, to_ndim, axis=0):
    x = _np.array(x)
    if x.ndim == to_ndim - 1:
        x = _np.expand_dims(x, axis=axis)

    if x.ndim != 0 and x.ndim < to_ndim:
        raise ValueError("The ndim was not adapted properly.")
    return x


def _get_wider_dtype(tensor_list):
    dtype_list = [_DTYPES.get(x.dtype, -1) for x in tensor_list]
    if len(dtype_list) == 1:
        return dtype_list[0], True

    wider_dtype_index = max(dtype_list)
    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]

    return wider_dtype, False


def convert_to_wider_dtype(tensor_list):
    wider_dtype, same = _get_wider_dtype(tensor_list)
    if same:
        return tensor_list

    return [cast(x, dtype=wider_dtype) for x in tensor_list]


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return _is_boolean(x[0])
    if isinstance(x, _np.ndarray):
        return x.dtype == bool
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if isinstance(x, _np.ndarray):
        return x.ndim > 0
    return False
