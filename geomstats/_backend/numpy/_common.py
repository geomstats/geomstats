import numpy as _np
from numpy import (
    array,
    eye,
    zeros,
)

atol = 1e-8
rtol = 1e-5


def is_array(x):
    return type(x) is _np.ndarray


def to_ndarray(x, to_ndim, axis=0, dtype=None):
    x = _np.asarray(x, dtype=dtype)

    if x.ndim > to_ndim:
        raise ValueError("The ndim cannot be adapted properly.")

    while x.ndim < to_ndim:
        x = _np.expand_dims(x, axis=axis)

    return x
