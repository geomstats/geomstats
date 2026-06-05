import autograd.numpy as _np

from geomstats._backend._dtype_utils import _pre_allow_complex_dtype

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


def cast(x, dtype):
    return x.astype(dtype)


def get_default_dtype():
    return as_dtype("float64")


_allow_complex_dtype = _pre_allow_complex_dtype(
    cast, _COMPLEX_DTYPES, get_default_dtype
)


def get_default_cdtype():
    return as_dtype("complex128")


def set_default_dtype(value):
    """Set backend default dtype.

    Parameters
    ----------
    value : str
        Possible values are ``float64``.
    """
    if value != "float64":
        raise ValueError("autograd only supports ``float64``")

    return get_default_dtype()


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
