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
    asarray,
    broadcast_arrays,
    broadcast_to,
    ceil,
    clip,
    complex64,
    complex128,
    concatenate,
    conj,
    copy,
    cos,
    cosh,
    cross,
    cumprod,
    cumsum,
    diag_indices,
    diagonal,
    einsum,
    empty,
    empty_like,
    equal,
    exp,
    expand_dims,
    flip,
    float32,
    float64,
    floor,
    greater,
    hsplit,
    hstack,
    inf,
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
    median,
    meshgrid,
    minimum,
    mod,
    moveaxis,
    ones,
    ones_like,
    pad,
    permute_dims,
    power,
    prod,
    quantile,
    ravel_multi_index,
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
    stack,
    std,
    sum,
    take,
    tan,
    tanh,
    tile,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    where,
    zeros_like,
)
from scipy.special import erf, gamma, polygamma  # NOQA

from . import (
    autodiff,  # NOQA
    linalg,  # NOQA
    random,  # NOQA
)
from ._common import (
    _is_boolean,
    _is_iterable,
    array,
    atol,
    eye,
    is_array,
    rtol,
    to_ndarray,
    zeros,
)
from ._dtype import (
    _get_wider_dtype,
    as_dtype,
    cast,
    convert_to_wider_dtype,
    get_default_cdtype,
    get_default_dtype,
    is_bool,
    is_complex,
    is_floating,
    set_default_dtype,
)

try:
    from numpy import trapezoid
except ImportError:
    from numpy import trapz as trapezoid


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return False


def to_numpy(x):
    return x


def from_numpy(x):
    return x


def imag(x):
    out = _np.imag(x)
    if hasattr(x, "dtype"):
        return out

    return array(out)


def real(x):
    out = _np.real(x)
    if hasattr(x, "dtype"):
        return out

    return array(out)


def squeeze(x, axis=None):
    if axis is None:
        return _np.squeeze(x)
    if x.shape[axis] != 1:
        return x
    return _np.squeeze(x, axis=axis)


def flatten(x):
    return x.flatten()


def transpose(x):
    """Return the transpose of a matrix.

    Parameters
    ----------
    x : array-like, shape=[..., n, m]
        Matrix.

    Returns
    -------
    transpose : array-like, shape=[..., n, m]
        Transposed matrix.
    """
    return _np.swapaxes(x, -1, -2)


def ndim(x):
    return x.ndim


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    if multiple_args:
        return _np.vectorize(pyfunc, signature=signature)(*x)
    return _np.vectorize(pyfunc, signature=signature)(x)


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
    data = array(data)
    out = zeros(target_shape, dtype=data.dtype)
    out.put(_np.ravel_multi_index(_np.array(indices).T, target_shape), data)
    return out


def vec_to_diag(vec):
    """Convert vector to diagonal matrix."""
    d = vec.shape[-1]
    return _np.squeeze(vec[..., None, :] * eye(d, dtype=vec.dtype)[None, :, :])


def tril_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = _np.tril_indices(n, k=k)
    return x[..., rows, cols]


def triu_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = _np.triu_indices(n, k=k)
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
    diag, tri_upp, tri_low = convert_to_wider_dtype([diag, tri_upp, tri_low])

    n = diag.shape[-1]
    (i,) = _np.diag_indices(n, ndim=1)
    j, k = _np.triu_indices(n, k=1)
    mat = zeros(diag.shape + (n,), dtype=diag.dtype)
    mat[..., i, i] = diag
    mat[..., j, k] = tri_upp
    mat[..., k, j] = tri_low
    return mat


def divide(a, b, ignore_div_zero=False):
    if ignore_div_zero is False:
        return _np.divide(a, b)

    wider_dtype, _ = _get_wider_dtype([a, b])
    return _np.divide(a, b, out=zeros(a.shape, dtype=wider_dtype), where=b != 0)


def matmul(*args, **kwargs):
    for arg in args:
        if arg.ndim == 1:
            raise ValueError("ndims must be >=2")
    return _np.matmul(*args, **kwargs)


def outer(a, b):
    if a.ndim > 1 and b.ndim > 1:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.multiply.outer(a, b)
    if b.ndim > 1:
        out = out.swapaxes(0, -2)

    return out


def matvec(A, b):
    if b.ndim == 1:
        return _np.matmul(A, b)
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


def scatter_add(input, dim, index, src):
    """Add values from src into input at the indices specified in index.

    Parameters
    ----------
    input : array-like
        Tensor to scatter values into.
    dim : int
        The axis along which to index.
    index : array-like
        The indices of elements to scatter.
    src : array-like
        The source element(s) to scatter.

    Returns
    -------
    input : array-like
        Modified input array.
    """
    if dim == 0:
        for i, val in zip(index, src):
            input[i] += val
        return input
    if dim == 1:
        for j in range(len(input)):
            for i, val in zip(index[j], src[j]):
                input[j, i] += float(val)
        return input
    raise NotImplementedError
