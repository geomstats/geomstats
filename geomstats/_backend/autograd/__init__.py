"""Autograd based computation backend."""

import autograd.numpy as _np
from autograd.numpy import (
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
    broadcast_to,
    ceil,
    clip,
    complex64,
    complex128,
    concatenate,
    conj,
    cos,
    cosh,
    cross,
    cumprod,
    cumsum,
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
    matrix_transpose,
    matvec,
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
    trapezoid,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    zeros_like,
)
from autograd.numpy.linalg import diagonal, vecdot
from autograd.scipy.special import erf, gamma, polygamma  # NOQA

from . import (
    _vjps,  # NOQA
    autodiff,  # NOQA
    linalg,  # NOQA
    random,  # NOQA
)
from ._common import (
    array,
    atol,
    expand_dims_to,
    eye,
    is_array,
    rtol,
    scatter_add,
    zeros,
)
from ._dtype import (
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


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def broadcast_arrays(*arrays):
    """Broadcast arrays against each other.

    Built on ``broadcast_to``, which autograd can differentiate.
    """
    target_shape = _np.broadcast_shapes(*[shape(array) for array in arrays])
    return tuple(broadcast_to(array, target_shape) for array in arrays)


def outer(a, b):
    """Compute the outer product of vectors.

    Parameters
    ----------
    a : array-like, shape=[..., n]
        Vector.
    b : array-like, shape=[..., m]
        Vector.

    Returns
    -------
    outer_ab : array-like, shape=[..., n, m]
        Outer product.
    """
    return a[..., :, None] * b[..., None, :]


def to_numpy(x):
    return x


def from_numpy(x):
    return x


def copy(x):
    return _np.array(x, copy=True)


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


def ndim(x):
    return x.ndim


def where(condition, x=None, y=None):
    """Return elements chosen from x or y depending on condition.

    Parameters
    ----------
    condition : array-like
        Boolean condition.
    x : array-like, optional
        Values selected where condition is true.
    y : array-like, optional
        Values selected where condition is false.

    Returns
    -------
    out : array-like or tuple of arrays
        Selected values if x and y are provided, otherwise matching indices.
    """
    if x is None and y is None:
        return _np.where(condition)

    return _np.where(array(condition), x, y)


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


def diag_embed(vec, offset=0):
    """Embed vectors as a diagonal of matrices, over the last two axes.

    It is the inverse of ``diagonal``, for the same offset.

    Parameters
    ----------
    vec : array-like, shape=[..., n]
        Vector.
    offset : int
        Offset of the diagonal from the main diagonal.

    Returns
    -------
    mat : array-like, shape=[..., n + |offset|, n + |offset|]
        Matrix, with the vector as its diagonal and zeros elsewhere.
    """
    n = vec.shape[-1]
    if offset == 0:
        return vec[..., None] * eye(n, dtype=vec.dtype)

    size = n + abs(offset)
    rows = eye(n, size, k=max(-offset, 0), dtype=vec.dtype)
    cols = eye(n, size, k=max(offset, 0), dtype=vec.dtype)
    return _np.matmul(_np.matrix_transpose(rows) * vec[..., None, :], cols)


def tril_to_vec(x, k=0):
    """Get the lower triangular part of square matrices as vectors.

    It is the inverse of ``tril_embed``, for the same ``k``. The entries are
    taken row by row.

    Parameters
    ----------
    x : array-like, shape=[..., n, n]
        Square matrix.
    k : int
        Diagonal bounding the triangular part, e.g. 0 to include the main
        diagonal and -1 to exclude it.

    Returns
    -------
    vec : array-like, shape=[..., m]
        Vector, with ``m`` the number of entries of the triangular part.
    """
    n = x.shape[-1]
    rows, cols = _np.tril_indices(n, k=k)
    return x[..., rows, cols]


def triu_to_vec(x, k=0):
    """Get the upper triangular part of square matrices as vectors.

    It is the inverse of ``triu_embed``, for the same ``k``. The entries are
    taken row by row.

    Parameters
    ----------
    x : array-like, shape=[..., n, n]
        Square matrix.
    k : int
        Diagonal bounding the triangular part, e.g. 0 to include the main
        diagonal and 1 to exclude it.

    Returns
    -------
    vec : array-like, shape=[..., m]
        Vector, with ``m`` the number of entries of the triangular part.
    """
    n = x.shape[-1]
    rows, cols = _np.triu_indices(n, k=k)
    return x[..., rows, cols]


def _tri_embed(vec, k, tri_indices):
    """Embed vectors as a triangular part of square matrices, with a gather.

    The vector is padded with a zero, at which the entries outside the
    triangular part point.
    """
    m = vec.shape[-1]
    size = int(round(((8 * m + 1) ** 0.5 - 1) / 2))
    if size * (size + 1) // 2 != m:
        raise ValueError(f"No triangular part of a square matrix has {m} entries.")

    n = size + (k if k > 0 else -k)
    rows, cols = tri_indices(n, k=k)
    index = _np.full((n, n), m)
    index[rows, cols] = _np.arange(m)

    zero = _np.zeros(vec.shape[:-1] + (1,), dtype=vec.dtype)
    return _np.concatenate([vec, zero], axis=-1)[..., index]


def tril_embed(vec, k=0):
    """Embed vectors as the lower triangular part of square matrices.

    It is the inverse of ``tril_to_vec``, for the same ``k``. The entries are
    placed row by row, and the rest of the matrix is zero.

    Parameters
    ----------
    vec : array-like, shape=[..., m]
        Vector, whose size is the number of entries of the triangular part.
    k : int
        Diagonal bounding the triangular part: 0 to include the main
        diagonal, -1 to exclude it. Requires ``k <= 0``.

    Returns
    -------
    mat : array-like, shape=[..., n, n]
        Matrix, with ``n`` such that the triangular part has ``m`` entries.
    """
    if k > 0:
        raise ValueError("tril_embed requires k <= 0.")
    return _tri_embed(vec, k, _np.tril_indices)


def triu_embed(vec, k=0):
    """Embed vectors as the upper triangular part of square matrices.

    It is the inverse of ``triu_to_vec``, for the same ``k``. The entries are
    placed row by row, and the rest of the matrix is zero.

    Parameters
    ----------
    vec : array-like, shape=[..., m]
        Vector, whose size is the number of entries of the triangular part.
    k : int
        Diagonal bounding the triangular part: 0 to include the main
        diagonal, 1 to exclude it. Requires ``k >= 0``.

    Returns
    -------
    mat : array-like, shape=[..., n, n]
        Matrix, with ``n`` such that the triangular part has ``m`` entries.
    """
    if k < 0:
        raise ValueError("triu_embed requires k >= 0.")
    return _tri_embed(vec, k, _np.triu_indices)


def divide(a, b, ignore_div_zero=False):
    if ignore_div_zero is False:
        return _np.divide(a, b)

    return _np.divide(
        a, b, out=zeros(a.shape, dtype=_np.result_type(a, b)), where=b != 0
    )


def matmul(*args, **kwargs):
    for arg in args:
        if arg.ndim == 1:
            raise ValueError("ndims must be >=2")
    return _np.matmul(*args, **kwargs)


def trace(x):
    """Compute the trace of square matrices, over the last two axes.

    Parameters
    ----------
    x : array-like, shape=[..., n, n]
        Square matrix.

    Returns
    -------
    trace : array-like, shape=[...]
        Trace.
    """
    return _np.linalg.trace(x)
