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
    broadcast_arrays,
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
    power,
    prod,
    quantile,
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
    transpose,
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

try:
    from autograd.numpy import trapezoid
except ImportError:
    from autograd.numpy import trapz as trapezoid

from autograd.scipy.special import erf, gamma, polygamma  # NOQA


from . import (
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
    _is_boolean,
    _is_iterable,
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


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def outer(a, b):
    if a.ndim > 1 and b.ndim > 1:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.outer(a, b).reshape(a.shape + b.shape)
    if b.ndim > 1:
        out = out.swapaxes(0, -2)

    return out


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


def one_hot(labels, num_classes):
    return eye(num_classes, dtype=_np.dtype("uint8"))[labels]


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


def ndim(x):
    return x.ndim


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


def ravel_tril_indices(n, k=0, m=None):
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = _np.tril_indices(n, k, m)
    return _np.ravel_multi_index(idxs, size)


def matmul(*args, **kwargs):
    for arg in args:
        if arg.ndim == 1:
            raise ValueError("ndims must be >=2")
    return _np.matmul(*args, **kwargs)


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
                if not isinstance(val, _np.float64):
                    val = float(val._value)
                input[j, i] += float(val)
        return input
    raise NotImplementedError
