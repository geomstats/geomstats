import numpy as _np

from .._backend_config import np_atol as atol
from .._backend_config import np_rtol as rtol


def to_ndarray(x, to_ndim, axis=0):
    x = _np.array(x)
    if x.ndim == to_ndim - 1:
        x = _np.expand_dims(x, axis=axis)

    if x.ndim != 0:
        if x.ndim < to_ndim:
            raise ValueError("The ndim was not adapted properly.")
    return x


def cast(x, dtype):
    return x.astype(dtype)
