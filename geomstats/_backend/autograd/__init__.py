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
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    zeros_like,
)
from autograd.numpy.linalg import vecdot

try:
    from autograd.numpy import trapezoid
except ImportError:
    from autograd.numpy import trapz as trapezoid

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
    eye,
    is_array,
    rtol,
    to_ndarray,
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


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    if multiple_args:
        return _np.vectorize(pyfunc, signature=signature)(*x)
    return _np.vectorize(pyfunc, signature=signature)(x)


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
                if not isinstance(val, _np.float64):
                    val = float(val._value)
                input[j, i] += float(val)
        return input
    raise NotImplementedError
