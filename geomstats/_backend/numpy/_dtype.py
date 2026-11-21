import numpy as _np

from geomstats._backend._dtype_utils import _pre_allow_complex_dtype

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
        raise ValueError("numpy only supports ``float64``")

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


def convert_to_wider_dtype(tensor_list):
    """Cast arrays to their common dtype, following numpy promotion rules.

    Parameters
    ----------
    tensor_list : list[array-like]
        Arrays.

    Returns
    -------
    tensor_list : list[array-like]
        Arrays, with the common dtype.
    """
    dtype = _np.result_type(*tensor_list)
    return [x if x.dtype == dtype else cast(x, dtype) for x in tensor_list]
