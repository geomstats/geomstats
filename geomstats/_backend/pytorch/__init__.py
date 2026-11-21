"""Pytorch based computation backend."""

from collections.abc import Iterable as _Iterable

import numpy as _np
import torch as _torch
from torch import (
    arange,
    argmin,
    asarray,
    clip,
    complex64,
    complex128,
    conj,
    empty,
    empty_like,
    erf,
    eye,
    flatten,
    float32,
    float64,
    greater,
    hstack,
    inf,
    int32,
    int64,
    isnan,
    less,
    logical_or,
    meshgrid,
    moveaxis,
    ones,
    ones_like,
    polygamma,
    reshape,
    scatter_add,
    searchsorted,
    stack,
    trapezoid,
    uint8,
    vstack,
    zeros,
    zeros_like,
)
from torch import broadcast_tensors as broadcast_arrays
from torch import repeat_interleave as repeat
from torch.special import gammaln as _gammaln

from . import (
    autodiff,  # NOQA
    linalg,  # NOQA
    random,  # NOQA
)
from ._common import array, atol, cast, from_numpy, rtol, to_numpy
from ._dtype import (
    _add_default_dtype_by_casting,
    _box_binary_scalar,
    _box_unary_scalar,
    _preserve_input_dtype,
    as_dtype,
    get_default_cdtype,
    get_default_dtype,
    is_bool,
    is_complex,
    is_floating,
    set_default_dtype,
)

abs = _box_unary_scalar(target=_torch.abs)
angle = _box_unary_scalar(target=_torch.angle)
arccos = _box_unary_scalar(target=_torch.arccos)
arccosh = _box_unary_scalar(target=_torch.arccosh)
arcsin = _box_unary_scalar(target=_torch.arcsin)
arctanh = _box_unary_scalar(target=_torch.arctanh)
ceil = _box_unary_scalar(target=_torch.ceil)
cos = _box_unary_scalar(target=_torch.cos)
cosh = _box_unary_scalar(target=_torch.cosh)
exp = _box_unary_scalar(target=_torch.exp)
floor = _box_unary_scalar(target=_torch.floor)
log = _box_unary_scalar(target=_torch.log)
real = _box_unary_scalar(target=_torch.real)
sign = _box_unary_scalar(target=_torch.sign)
sin = _box_unary_scalar(target=_torch.sin)
sinh = _box_unary_scalar(target=_torch.sinh)
sqrt = _box_unary_scalar(target=_torch.sqrt)
tan = _box_unary_scalar(target=_torch.tan)
tanh = _box_unary_scalar(target=_torch.tanh)


arctan2 = _box_binary_scalar(target=_torch.atan2)
mod = _box_binary_scalar(target=_torch.remainder, box_x2=False)
power = _box_binary_scalar(target=_torch.pow, box_x2=False)


std = _preserve_input_dtype(_add_default_dtype_by_casting(target=_torch.std))


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def matmul(x, y, out=None):
    for array_ in [x, y]:
        if array_.ndim == 1:
            raise ValueError("ndims must be >=2")

    return _torch.matmul(x, y, out=out)


def argmax(a, **kwargs):
    if a.dtype == _torch.bool:
        a = a.to(_torch.uint8)
    return _torch.argmax(a, **kwargs)


def less_equal(x, y, **kwargs):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    return _torch.le(x, y, **kwargs)


def split(x, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, int):
        indices_or_sections = x.shape[axis] // indices_or_sections
        return _torch.split(x, indices_or_sections, dim=axis)
    indices_or_sections = _np.asarray(indices_or_sections)
    intervals_length = indices_or_sections[1:] - indices_or_sections[:-1]
    last_interval_length = x.shape[axis] - indices_or_sections[-1]
    if last_interval_length > 0:
        intervals_length = _np.append(intervals_length, last_interval_length)
    intervals_length = _np.insert(intervals_length, 0, indices_or_sections[0])
    return _torch.split(x, tuple(intervals_length), dim=axis)


def logical_and(x, y):
    if _torch.is_tensor(x) or _torch.is_tensor(y):
        return _torch.logical_and(_torch.as_tensor(x), _torch.as_tensor(y))
    return x and y


def any(x, axis=None):
    x = _torch.as_tensor(x).bool()
    if axis is None:
        return _torch.any(x)
    return _torch.any(x, dim=axis)


def flip(x, axis=None):
    if isinstance(axis, int):
        axis = [axis]
    if axis is None:
        axis = list(range(x.ndim))
    return _torch.flip(x, dims=axis)


def concatenate(seq, axis=0, out=None):
    return _torch.cat(seq, dim=axis, out=out)


def all(x, axis=None):
    x = _torch.as_tensor(x).bool()
    if axis is None:
        return _torch.all(x)
    return _torch.all(x, dim=axis)


def allclose(a, b, atol=atol, rtol=rtol):
    if not isinstance(a, _torch.Tensor):
        a = _torch.tensor(a)
    if not isinstance(b, _torch.Tensor):
        b = _torch.tensor(b)
    a, b = _torch.broadcast_tensors(a, b)
    return _torch.allclose(a, b, atol=atol, rtol=rtol)


def shape(val):
    if not is_array(val):
        val = array(val)
    return val.shape


def amax(a, axis=None):
    if axis is None:
        return _torch.max(array(a))
    return _torch.amax(array(a), dim=axis)


def maximum(a, b):
    return _torch.max(array(a), array(b))


def minimum(a, b):
    return _torch.min(array(a), array(b))


def mean(x, axis=None, dtype=None, keepdims=False, out=None):
    return _torch.mean(x, dim=axis, dtype=dtype, keepdim=keepdims, out=out)


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
    if x.dim() > ndim:
        raise ValueError(f"Cannot expand an array with {x.dim()} axes to {ndim}.")

    while x.dim() < ndim:
        x = _torch.unsqueeze(x, dim=axis)

    return x


def broadcast_to(x, shape):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    return x.expand(shape)


def isclose(x, y, rtol=rtol, atol=atol):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    return _torch.isclose(x, y, atol=atol, rtol=rtol)


def sum(x, axis=None, keepdims=None, dtype=None):
    if axis is None:
        if keepdims is None:
            return _torch.sum(x, dtype=dtype)
        return _torch.sum(x, keepdim=keepdims, dtype=dtype)
    if keepdims is None:
        return _torch.sum(x, dim=axis, dtype=dtype)
    return _torch.sum(x, dim=axis, keepdim=keepdims, dtype=dtype)


def einsum(equation, *inputs):
    input_tensors_list = [arg if is_array(arg) else array(arg) for arg in inputs]
    return _torch.einsum(equation, *input_tensors_list)


def matrix_transpose(x):
    """Transpose the last two axes of an array.

    Parameters
    ----------
    x : array-like, shape=[..., n, m]
        Matrix.

    Returns
    -------
    x_t : array-like, shape=[..., m, n]
        Transposed matrix.
    """
    return x.mT


def permute_dims(x, axes=None):
    if axes is None:
        axes = tuple(range(x.ndim - 1, -1, -1))

    return _torch.permute(x, axes)


def squeeze(x, axis=None):
    if not is_array(x):
        return x
    if axis is None:
        return _torch.squeeze(x)
    return _torch.squeeze(x, dim=axis)


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
    return _torch.sum(_torch.linalg.diagonal(x), dim=-1)


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    start_is_array = _torch.is_tensor(start)
    stop_is_array = _torch.is_tensor(stop)

    if not (start_is_array or stop_is_array) and endpoint:
        return _torch.linspace(start=start, end=stop, steps=num, dtype=dtype)

    if not start_is_array:
        start = _torch.tensor(start)
    if not stop_is_array:
        stop = _torch.tensor(stop)
    start, stop = _torch.broadcast_tensors(start, stop)
    result_shape = (num, *start.shape)
    start = _torch.flatten(start)
    stop = _torch.flatten(stop)

    if endpoint:
        result = _torch.vstack(
            [
                _torch.linspace(start=start[i], end=stop[i], steps=num, dtype=dtype)
                for i in range(start.shape[0])
            ]
        ).T
    else:
        result = _torch.vstack(
            [
                _torch.arange(
                    start=start[i],
                    end=stop[i],
                    step=(stop[i] - start[i]) / num,
                    dtype=dtype,
                )
                for i in range(start.shape[0])
            ]
        ).T

    return _torch.reshape(result, result_shape)


def equal(a, b, **kwargs):
    if not is_array(a):
        a = array(a)

    if not is_array(b):
        b = array(b)
    return _torch.eq(a, b, **kwargs)


def tril(mat, k=0):
    return _torch.tril(mat, diagonal=k)


def triu(mat, k=0):
    return _torch.triu(mat, diagonal=k)


def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    return _torch.tril_indices(row=n, col=m, offset=k)


def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    return _torch.triu_indices(row=n, col=m, offset=k)


def tile(x, y):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    return x.repeat(y)


def expand_dims(x, axis=0):
    return _torch.unsqueeze(x, dim=axis)


def ndim(x):
    return x.dim()


def hsplit(x, indices_or_section):
    if isinstance(indices_or_section, int):
        indices_or_section = x.shape[-1] // indices_or_section
    return _torch.split(x, indices_or_section, dim=-1)


def diagonal(x, offset=0):
    """Get the diagonal of matrices, over the last two axes.

    Parameters
    ----------
    x : array-like, shape=[..., n, m]
        Matrix.
    offset : int
        Offset of the diagonal from the main diagonal.

    Returns
    -------
    diag : array-like, shape=[..., k]
        Diagonal.
    """
    return _torch.diagonal(x, offset=offset, dim1=-2, dim2=-1)


def prod(x, axis=None):
    if axis is None:
        return _torch.prod(x)
    return _torch.prod(x, axis)


def where(condition, x=None, y=None):
    if not _torch.is_tensor(condition):
        condition = array(condition)

    if x is None and y is None:
        return _torch.where(condition)
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    return _torch.where(condition, x, y)


def copy(x):
    return x.clone()


def cumsum(x, axis=None, dtype=None):
    if not _torch.is_tensor(x):
        x = array(x, dtype=dtype)
    if axis is None:
        return x.flatten().cumsum(dim=0, dtype=dtype)
    return _torch.cumsum(x, dim=axis, dtype=dtype)


def cumprod(x, axis=None, dtype=None):
    if axis is None:
        return x.flatten().cumprod(dim=0, dtype=dtype)
    return _torch.cumprod(x, axis, dtype=dtype)


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
    return _torch.sparse_coo_tensor(
        _torch.LongTensor(indices).t(),
        array(data),
        _torch.Size(target_shape),
    ).to_dense()


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
    return _torch.diag_embed(vec, offset=offset)


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
    rows, cols = tril_indices(n, k=k)
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
    rows, cols = triu_indices(n, k=k)
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
    rows, cols = tri_indices(n, n, offset=k)
    index = _torch.full((n, n), m, dtype=_torch.long)
    index[rows, cols] = _torch.arange(m)

    zero = _torch.zeros(vec.shape[:-1] + (1,), dtype=vec.dtype, device=vec.device)
    return _torch.cat([vec, zero], dim=-1)[..., index.to(vec.device)]


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
    return _tri_embed(vec, k, _torch.tril_indices)


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
    return _tri_embed(vec, k, _torch.triu_indices)


def divide(a, b, ignore_div_zero=False):
    if ignore_div_zero is False:
        return _torch.divide(a, b)
    quo = _torch.divide(a, b)
    return _torch.nan_to_num(quo, nan=0.0, posinf=0.0, neginf=0.0)


def sort(a, axis=-1):
    sorted_a, _ = _torch.sort(a, dim=axis)
    return sorted_a


def amin(a, axis=None):
    if axis is None:
        return _torch.min(array(a))
    return _torch.amin(array(a), dim=axis)


def take(a, indices, axis=None):
    """Take elements of an array along an axis, as ``np.take``.

    Parameters
    ----------
    a : array-like
        Array.
    indices : array-like or int
        Indices of the elements to take.
    axis : int
        Axis along which to take the elements. If None, the array is
        flattened first.

    Returns
    -------
    out : array-like, shape=[*a.shape[:axis], *indices.shape, *a.shape[axis + 1:]]
        Elements of the array.
    """
    indices = _torch.as_tensor(indices, device=a.device)
    if axis is None:
        a = _torch.reshape(a, (-1,))
        axis = 0
    axis = axis % a.ndim

    out = _torch.index_select(a, axis, _torch.reshape(indices, (-1,)))
    return _torch.reshape(out, a.shape[:axis] + indices.shape + a.shape[axis + 1 :])


def _unnest_iterable(ls):
    out = []
    if isinstance(ls, _Iterable):
        for inner_ls in ls:
            out.extend(_unnest_iterable(inner_ls))
    else:
        out.append(ls)

    return out


def is_array(x):
    return _torch.is_tensor(x)


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


def matvec(A, b):
    """Compute the matrix-vector product.

    Parameters
    ----------
    A : array-like, shape=[..., n, m]
        Matrix.
    b : array-like, shape=[..., m]
        Vector.

    Returns
    -------
    Ab : array-like, shape=[..., n]
        Matrix-vector product.
    """
    if b.ndim == 1:
        return _torch.matmul(A, b)

    return _torch.matmul(A, b[..., None])[..., 0]


def vecdot(a, b, axis=-1):
    """Compute the dot product of vectors, conjugating the first one.

    Parameters
    ----------
    a : array-like, shape=[..., n]
        Vector.
    b : array-like, shape=[..., n]
        Vector.
    axis : int
        Axis of the vectors.

    Returns
    -------
    vecdot_ab : array-like, shape=[...,]
        Dot product.
    """
    return _torch.linalg.vecdot(a, b, dim=axis)


def cross(a, b):
    if a.shape != b.shape:
        a, b = broadcast_arrays(a, b)
    return _torch.cross(a, b, dim=-1)


def gamma(a):
    return _torch.exp(_gammaln(a))


def imag(a):
    if not _torch.is_tensor(a):
        a = _torch.tensor(a)
    if is_complex(a):
        return _torch.imag(a)
    return _torch.zeros(a.shape, dtype=a.dtype)


def unique(ar, axis=None):
    return _torch.unique(ar, dim=axis)
