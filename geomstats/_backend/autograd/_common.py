import autograd.numpy as _np
from autograd.extend import primitive as _primitive
from autograd.numpy import (
    array,
    eye,
    zeros,
)

from ..numpy import scatter_add as _numpy_scatter_add

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


_scatter_add = _primitive(_numpy_scatter_add)


def scatter_add(input, dim, index, src):
    """Add values of src into input, at the indices of index along an axis.

    Same as the numpy backend's, as an autograd primitive, whose VJP is
    registered in ``_vjps``. The arguments are passed positionally, as
    autograd only differentiates positional arguments.
    """
    return _scatter_add(input, dim, index, src)
