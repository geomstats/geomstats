"""Numpy based computation backend."""

import numpy as _np
from numpy import (
    abs,
    all,
    allclose,
    amax,
    amin,
    angle,
    any,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    arctanh,
    argmax,
    argmin,
    array,
    broadcast_arrays,
    broadcast_to,
    ceil,
    clip,
    concatenate,
    conj,
    cos,
    cosh,
    cross,
    cumprod,
    cumsum,
    diag_indices,
    diagonal,
)
from numpy import dtype as _ndtype  # NOQA
from numpy import (
    einsum,
    empty,
    empty_like,
    equal,
    exp,
    expand_dims,
    eye,
    flip,
    float32,
    float64,
    floor,
    greater,
    hsplit,
    hstack,
    imag,
    int32,
    int64,
    isclose,
    isnan,
    kron,
    less,
    less_equal,
    linspace,
    log,
    logical_and,
    logical_or,
    maximum,
    mean,
    meshgrid,
    minimum,
    mod,
    moveaxis,
    ones,
    ones_like,
    pad,
    power,
    prod,
    real,
    repeat,
    reshape,
    searchsorted,
    shape,
    sign,
    sin,
    sinh,
    sort,
    split,
    sqrt,
    squeeze,
    stack,
    std,
    sum,
    take,
    tan,
    tanh,
    tile,
    transpose,
    trapz,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    where,
    zeros,
    zeros_like,
)
from scipy.sparse import coo_matrix as _coo_matrix  # NOQA
from scipy.special import erf, polygamma  # NOQA

from .._backend_config import np_atol as atol
from .._backend_config import np_rtol as rtol
from . import autodiff  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA
from ._common import to_ndarray

_DTYPES = {
    _ndtype("int32"): 0,
    _ndtype("int64"): 1,
    _ndtype("float32"): 2,
    _ndtype("float64"): 3,
    _ndtype("complex64"): 4,
    _ndtype("complex128"): 5,
}


def to_numpy(x):
    return x


def from_numpy(x):
    return x


def convert_to_wider_dtype(tensor_list):
    dtype_list = [_DTYPES.get(x.dtype, -1) for x in tensor_list]
    if len(set(dtype_list)) == 1:
        return tensor_list

    wider_dtype_index = max(dtype_list)

    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return tensor_list


def flatten(x):
    return x.flatten()


def one_hot(labels, num_classes):
    return _np.eye(num_classes, dtype=_np.dtype("uint8"))[labels]


def get_mask_i_float(i, n):
    """Create a 1D array of zeros with one element at one, with floating type.

    Parameters
    ----------
    i : int
        Index of the non-zero element.
    n: n
        Length of the created array.

    Returns
    -------
    mask_i_float : array-like, shape=[n,]
        1D array of zeros except at index i, where it is one
    """
    range_n = arange(n)
    i_float = cast(array([i]), int32)[0]
    mask_i = equal(range_n, i_float)
    mask_i_float = cast(mask_i, float32)
    return mask_i_float


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
        return ndim(x) > 0
    return False


def assignment(x, values, indices, axis=0):
    """Assign values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)

    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] = values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    len_indices = len(indices) if _is_iterable(indices) else 1
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        if not zip_indices:
            len_indices = len(indices) if _is_iterable(indices) else 1
        len_values = len(values) if _is_iterable(values) else 1
        if len_values > 1 and len_values != len_indices:
            raise ValueError("Either one value or as many values as indices")
        x_new[indices] = values
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] = values
    return x_new


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    values : {float, list(float)}
        Value or list of values to be assigned.
    indices : {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)

    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] += values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        len_indices = len(indices) if _is_iterable(indices) else 1
        len_values = len(values) if _is_iterable(values) else 1
        if len_values > 1 and len_values != len_indices:
            raise ValueError("Either one value or as many values as indices")
        x_new[indices] += values
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] += values
    return x_new


def get_slice(x, indices):
    """Return a slice of an array, following Numpy's style.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    indices : iterable(iterable(int))
        Indices which are kept along each axis, starting from 0.

    Returns
    -------
    slice : array-like
        Slice of x given by indices.

    Notes
    -----
    This follows Numpy's convention: indices are grouped by axis.

    Examples
    --------
    >>> a = np.array(range(30)).reshape(3,10)
    >>> get_slice(a, ((0, 2), (8, 9)))
    array([8, 29])
    """
    return x[indices]


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    if multiple_args:
        return _np.vectorize(pyfunc, signature=signature)(*x)
    return _np.vectorize(pyfunc, signature=signature)(x)


def cast(x, dtype):
    return x.astype(dtype)


def set_diag(x, new_diag):
    """Set the diagonal along the last two axis.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    new_diag : array-like, shape=[dim[-2]]
        Values to set on the diagonal.

    Returns
    -------
    None

    Notes
    -----
    This mimics tensorflow.linalg.set_diag(x, new_diag), when new_diag is a
    1-D array, but modifies x instead of creating a copy.
    """
    arr_shape = x.shape
    x[..., range(arr_shape[-2]), range(arr_shape[-1])] = new_diag
    return x


def ndim(x):
    return x.ndim


def copy(x):
    return x.copy()


def array_from_sparse(indices, data, target_shape):
    """Create an array of given shape, with values at specific indices.

    The rest of the array will be filled with zeros.

    Parameters
    ----------
    indices : iterable(tuple(int))
        Index of each element which will be assigned a specific value.
    data : iterable(scalar)
        Value associated at each index.
    target_shape : tuple(int)
        Shape of the output array.

    Returns
    -------
    a : array, shape=target_shape
        Array of zeros with specified values assigned to specified indices.
    """
    return array(_coo_matrix((data, list(zip(*indices))), target_shape).todense())


def vec_to_diag(vec):
    """Convert vector to diagonal matrix."""
    d = vec.shape[-1]
    return _np.squeeze(vec[..., None, :] * _np.eye(d)[None, :, :])


def tril_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = tril_indices(n, k=k)
    return x[..., rows, cols]


def triu_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = triu_indices(n, k=k)
    return x[..., rows, cols]


def mat_from_diag_triu_tril(diag, tri_upp, tri_low):
    """Build matrix from given components.

    Forms a matrix from diagonal, strictly upper triangular and
    strictly lower traingular parts.

    Parameters
    ----------
    diag : array_like, shape=[..., n]
    tri_upp : array_like, shape=[..., (n * (n - 1)) / 2]
    tri_low : array_like, shape=[..., (n * (n - 1)) / 2]

    Returns
    -------
    mat : array_like, shape=[..., n, n]
    """
    n = diag.shape[-1]
    (i,) = _np.diag_indices(n, ndim=1)
    j, k = _np.triu_indices(n, k=1)
    mat = _np.zeros(diag.shape + (n,))
    mat[..., i, i] = diag
    mat[..., j, k] = tri_upp
    mat[..., k, j] = tri_low
    return mat


def divide(a, b, ignore_div_zero=False):
    if ignore_div_zero is False:
        return _np.divide(a, b)
    return _np.divide(a, b, out=_np.zeros_like(a), where=b != 0)


def ravel_tril_indices(n, k=0, m=None):
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = _np.tril_indices(n, k, m)
    return _np.ravel_multi_index(idxs, size)


def is_array(x):
    return type(x) is _np.ndarray


def matmul(*args, **kwargs):
    for arg in args:
        if arg.ndim == 1:
            raise ValueError("ndims must be >=2")
    return _np.matmul(*args, **kwargs)


def outer(a, b):
    if a.ndim == 2 and b.ndim == 2:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.multiply.outer(a, b)
    if b.ndim == 2:
        out = out.swapaxes(-3, -2)

    return out


def matvec(A, b):
    if b.ndim == 1:
        return _np.matmul(A, b)
    else:
        if A.ndim == 2:
            return _np.matmul(A, b.T).T
        return _np.einsum("...ij,...j->...i", A, b)


def dot(a, b):
    if b.ndim == 1:
        return _np.dot(a, b)

    if a.ndim == 1:
        return _np.dot(a, b.T)

    return _np.einsum("...i,...i->...", a, b)


def trace(a):
    return _np.trace(a, axis1=-2, axis2=-1)
