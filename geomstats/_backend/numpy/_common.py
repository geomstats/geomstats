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


def expand_dims_to(x, ndim, axis=0):
    """Insert axes at ``axis`` until an array has ``ndim`` axes.

    Parameters
    ----------
    x : array-like
        Array, with at most ``ndim`` axes.
    ndim : int
        Number of axes of the output.
    axis : int
        Position of the inserted axes.

    Returns
    -------
    x : array-like
        Array with ``ndim`` axes.
    """
    if x.ndim > ndim:
        raise ValueError(f"Cannot expand an array with {x.ndim} axes to {ndim}.")

    while x.ndim < ndim:
        x = _np.expand_dims(x, axis=axis)

    return x
