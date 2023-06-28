"""Pytorch based computation backend."""

from collections.abc import Iterable as _Iterable

import numpy as _np
import torch as _torch
from torch import arange, argmin
from torch import broadcast_tensors as broadcast_arrays
from torch import (
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
    int32,
    int64,
    isnan,
    kron,
    less,
    logical_or,
    mean,
    meshgrid,
    moveaxis,
    ones,
    ones_like,
    polygamma,
    quantile,
)
from torch import repeat_interleave as repeat
from torch import (
    reshape,
    scatter_add,
    stack,
    trapz,
    uint8,
    unique,
    vstack,
    zeros,
    zeros_like,
)
from torch.special import gammaln as _gammaln

from .._backend_config import pytorch_atol as atol
from .._backend_config import pytorch_rtol as rtol
from . import autodiff  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA
from ._common import array, cast, from_numpy
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

_DTYPES = {
    int32: 0,
    int64: 1,
    float32: 2,
    float64: 3,
    complex64: 4,
    complex128: 5,
}


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


searchsorted = _raise_not_implemented_error


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
mod = _box_binary_scalar(target=_torch.fmod, box_x2=False)
power = _box_binary_scalar(target=_torch.pow, box_x2=False)


std = _preserve_input_dtype(_add_default_dtype_by_casting(target=_torch.std))


def matmul(x, y, out=None):
    for array_ in [x, y]:
        if array_.ndim == 1:
            raise ValueError("ndims must be >=2")

    x, y = convert_to_wider_dtype([x, y])
    return _torch.matmul(x, y, out=out)


def to_numpy(x):
    return x.numpy()


def one_hot(labels, num_classes):
    if not _torch.is_tensor(labels):
        labels = _torch.LongTensor(labels)
    return _torch.nn.functional.one_hot(labels, num_classes).type(_torch.uint8)


def argmax(a, **kwargs):
    if a.dtype == _torch.bool:
        return _torch.as_tensor(_np.argmax(a.data.numpy(), **kwargs))
    return _torch.argmax(a, **kwargs)


def convert_to_wider_dtype(tensor_list):
    dtype_list = [_DTYPES.get(x.dtype, -1) for x in tensor_list]
    if len(set(dtype_list)) == 1:
        return tensor_list

    wider_dtype_index = max(dtype_list)

    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return tensor_list


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
    indices_or_sections = _np.array(indices_or_sections)
    intervals_length = indices_or_sections[1:] - indices_or_sections[:-1]
    last_interval_length = x.shape[axis] - indices_or_sections[-1]
    if last_interval_length > 0:
        intervals_length = _np.append(intervals_length, last_interval_length)
    intervals_length = _np.insert(intervals_length, 0, indices_or_sections[0])
    return _torch.split(x, tuple(intervals_length), dim=axis)


def logical_and(x, y):
    if _torch.is_tensor(x) or _torch.is_tensor(y):
        return x * y
    return x and y


def any(x, axis=None):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if axis is None:
        return _torch.any(x)
    if isinstance(axis, int):
        return _torch.any(x.bool(), axis)
    if len(axis) == 1:
        return _torch.any(x, *axis)
    axis = list(axis)
    for i_axis, one_axis in enumerate(axis):
        if one_axis < 0:
            axis[i_axis] = ndim(x) + one_axis
    new_axis = tuple(k - 1 if k >= 0 else k for k in axis[1:])
    return any(_torch.any(x.bool(), axis[0]), new_axis)


def flip(x, axis):
    if isinstance(axis, int):
        axis = [axis]
    if axis is None:
        axis = list(range(x.ndim))
    return _torch.flip(x, dims=axis)


def concatenate(seq, axis=0, out=None):
    seq = convert_to_wider_dtype(seq)
    return _torch.cat(seq, dim=axis, out=out)


def all(x, axis=None):
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if axis is None:
        return x.bool().all()
    if isinstance(axis, int):
        return _torch.all(x.bool(), axis)
    if len(axis) == 1:
        return _torch.all(x, *axis)
    axis = list(axis)
    for i_axis, one_axis in enumerate(axis):
        if one_axis < 0:
            axis[i_axis] = ndim(x) + one_axis
    new_axis = tuple(k - 1 if k >= 0 else k for k in axis[1:])
    return all(_torch.all(x.bool(), axis[0]), new_axis)


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
    >>> a = torch.tensor(range(30)).reshape(3,10)
    >>> get_slice(a, ((0, 2), (8, 9)))
    tensor([8, 29])
    """
    return x[indices]


def allclose(a, b, atol=atol, rtol=rtol):
    if not isinstance(a, _torch.Tensor):
        a = _torch.tensor(a)
    if not isinstance(b, _torch.Tensor):
        b = _torch.tensor(b)
    a = to_ndarray(a.float(), to_ndim=1)
    b = to_ndarray(b.float(), to_ndim=1)
    n_a = a.shape[0]
    n_b = b.shape[0]
    nb_dim = a.dim()
    if n_a > n_b:
        reps = (int(n_a / n_b),) + (nb_dim - 1) * (1,)
        b = tile(b, reps)
    elif n_a < n_b:
        reps = (int(n_b / n_a),) + (nb_dim - 1) * (1,)
        a = tile(a, reps)
    return _torch.allclose(a, b, atol=atol, rtol=rtol)


def shape(val):
    if not is_array(val):
        val = array(val)
    return val.shape


def amax(a, axis=None):
    if axis is None:
        return _torch.max(array(a))
    return _torch.max(array(a), dim=axis).values


def maximum(a, b):
    return _torch.max(array(a), array(b))


def minimum(a, b):
    return _torch.min(array(a), array(b))


def to_ndarray(x, to_ndim, axis=0):
    if not _torch.is_tensor(x):
        x = array(x)

    if x.dim() == to_ndim - 1:
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
    input_tensors_list = convert_to_wider_dtype(input_tensors_list)

    return _torch.einsum(equation, *input_tensors_list)


def transpose(x, axes=None):
    if axes:
        return x.permute(axes)
    if x.dim() == 1:
        return x
    if x.dim() > 2 and axes is None:
        return x.permute(tuple(range(x.ndim)[::-1]))
    return x.t()


def squeeze(x, axis=None):
    if not is_array(x):
        return x
    if axis is None:
        return _torch.squeeze(x)
    return _torch.squeeze(x, dim=axis)


def trace(x):
    if x.ndim == 2:
        return _torch.trace(x)

    return _torch.einsum("...ii", x)


def linspace(start, stop, num=50, dtype=None):
    start_is_array = _torch.is_tensor(start)
    stop_is_array = _torch.is_tensor(stop)

    if not (start_is_array or stop_is_array):
        return _torch.linspace(start=start, end=stop, steps=num, dtype=dtype)

    if not start_is_array:
        start = _torch.tensor(start)
    if not stop_is_array:
        stop = _torch.tensor(stop)
    start, stop = _torch.broadcast_tensors(start, stop)
    result_shape = (num, *start.shape)
    start = _torch.flatten(start)
    stop = _torch.flatten(stop)

    return _torch.reshape(
        _torch.vstack(
            [
                _torch.linspace(start=start[i], end=stop[i], steps=num, dtype=dtype)
                for i in range(start.shape[0])
            ]
        ).T,
        result_shape,
    )


def equal(a, b, **kwargs):
    if not is_array(a):
        a = array(a)

    if not is_array(b):
        b = array(b)
    return _torch.eq(a, b, **kwargs)


def diag_indices(*args, **kwargs):
    return tuple(map(_torch.from_numpy, _np.diag_indices(*args, **kwargs)))


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


def diagonal(x, offset=0, axis1=0, axis2=1):
    return _torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


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
    off_diag = (1 - _torch.eye(arr_shape[-1])) * x
    diag = _torch.einsum("ij,...i->...ij", _torch.eye(new_diag.shape[-1]), new_diag)
    return diag + off_diag


def prod(x, axis=None):
    if axis is None:
        axis = 0
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
    y = cast(y, x.dtype)
    return _torch.where(condition, x, y)


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return _is_boolean(x[0])
    if _torch.is_tensor(x):
        return x.dtype in [_torch.bool, _torch.uint8]
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if _torch.is_tensor(x):
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
    values = array(values)
    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] += values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    if zip_indices:
        indices = list(zip(*indices))
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
        axis = 0
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
    return _torch.sparse.FloatTensor(
        _torch.LongTensor(indices).t(),
        array(data),
        _torch.Size(target_shape),
    ).to_dense()


def vectorize(x, pyfunc, multiple_args=False, **kwargs):
    if multiple_args:
        return stack(list(map(lambda y: pyfunc(*y), zip(*x))))
    return stack(list(map(pyfunc, x)))


def vec_to_diag(vec):
    return _torch.diag_embed(vec, offset=0)


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
    diag, tri_upp, tri_low = convert_to_wider_dtype([diag, tri_upp, tri_low])

    n = diag.shape[-1]
    (i,) = diag_indices(n, ndim=1)
    j, k = triu_indices(n, k=1)
    mat = _torch.zeros((diag.shape + (n,)), dtype=diag.dtype)
    mat[..., i, i] = diag
    mat[..., j, k] = tri_upp
    mat[..., k, j] = tri_low
    return mat


def divide(a, b, ignore_div_zero=False):
    if ignore_div_zero is False:
        return _torch.divide(a, b)
    quo = _torch.divide(a, b)
    return _torch.nan_to_num(quo, nan=0.0, posinf=0.0, neginf=0.0)


def ravel_tril_indices(n, k=0, m=None):
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = _np.tril_indices(n, k, m)
    return _torch.from_numpy(_np.ravel_multi_index(idxs, size))


def sort(a, axis=-1):
    sorted_a, _ = _torch.sort(a, dim=axis)
    return sorted_a


def amin(a, axis=-1):
    (values, _) = _torch.min(a, dim=axis)
    return values


def take(a, indices, axis=0):
    if not _torch.is_tensor(indices):
        indices = _torch.as_tensor(indices)

    return _torch.squeeze(_torch.index_select(a, axis, indices))


def _unnest_iterable(ls):
    out = []
    if isinstance(ls, _Iterable):
        for inner_ls in ls:
            out.extend(_unnest_iterable(inner_ls))
    else:
        out.append(ls)

    return out


def pad(a, pad_width, constant_value=0.0):
    return _torch.nn.functional.pad(
        a, _unnest_iterable(reversed(pad_width)), value=constant_value
    )


def is_array(x):
    return _torch.is_tensor(x)


def outer(a, b):
    # TODO: improve for torch > 1.9 (dims=0 fails in 1.9)
    return _torch.einsum("...i,...j->...ij", a, b)


def matvec(A, b):
    A, b = convert_to_wider_dtype([A, b])

    if A.ndim == 2 and b.ndim == 1:
        return _torch.mv(A, b)

    if b.ndim == 1:  # A.ndim > 2
        return _torch.matmul(A, b)

    if A.ndim == 2:  # b.ndim > 1
        return _torch.matmul(A, b.T).T

    return _torch.einsum("...ij,...j->...i", A, b)


def dot(a, b):
    a, b = convert_to_wider_dtype([a, b])

    if a.ndim == 1 and b.ndim == 1:
        return _torch.dot(a, b)

    if b.ndim == 1:
        return _torch.tensordot(a, b, dims=1)

    if a.ndim == 1:
        return _torch.tensordot(a, b.T, dims=1)

    return _torch.einsum("...i,...i->...", a, b)


def cross(a, b):
    if a.ndim + b.ndim == 3 or a.ndim == b.ndim == 2 and a.shape[0] != b.shape[0]:
        a, b = broadcast_arrays(a, b)
    return _torch.cross(*convert_to_wider_dtype([a, b]))


def gamma(a):
    return _torch.exp(_gammaln(a))


def imag(a):
    if not _torch.is_tensor(a):
        a = _torch.tensor(a)
    if is_complex(a):
        return _torch.imag(a)
    return _torch.zeros(a.shape, dtype=a.dtype)
