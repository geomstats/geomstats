"""Pytorch based computation backend."""

from functools import wraps

import numpy as _np
import torch
from torch import (  # NOQA
    abs,
    acos as arccos,
    arange,
    argmax,
    argmin,
    asin as arcsin,
    atan2 as arctan2,
    ceil,
    clamp as clip,
    cos,
    cosh,
    div as divide,
    empty_like,
    eq,
    exp,
    eye,
    flatten,
    float32,
    float64,
    floor,
    fmod as mod,
    ger as outer,
    gt as greater,
    int32,
    int64,
    isnan,
    le as less_equal,
    log,
    lt as less,
    matmul,
    meshgrid,
    nonzero,
    ones,
    ones_like,
    pow as power,
    repeat_interleave as repeat,
    reshape,
    sign,
    sin,
    sinh,
    stack,
    std,
    tan,
    tanh,
    tril,
    where,
    zeros,
    zeros_like
)

from . import linalg  # NOQA
from . import random  # NOQA


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


searchsorted = _raise_not_implemented_error
vectorize = _raise_not_implemented_error


def _box_scalar(function):
    @wraps(function)
    def wrapper(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return function(x)
    return wrapper


cos = _box_scalar(cos)
cosh = _box_scalar(cosh)
exp = _box_scalar(exp)
log = _box_scalar(log)
sin = _box_scalar(sin)
sinh = _box_scalar(sinh)


def empty(shape, dtype=float64):
    return torch.empty(*shape, dtype=dtype)


def split(ary, indices_or_sections, axis=0):
    return torch.split(ary, indices_or_sections, dim=axis)


def logical_or(x, y):
    return x or y


def logical_and(x, y):
    if torch.is_tensor(x):
        return x & y
    return x and y


def any(x, axis=None):
    if axis is None and torch.is_tensor(x):
        return x.bool().any()
    numpy_result = _np.array(_np.any(_np.array(x), axis=axis))
    return torch.from_numpy(numpy_result)


def cast(x, dtype):
    return array(x).to(dtype)


def flip(x, axis):
    if isinstance(axis, int):
        axis = [axis]
    if axis is None:
        axis = list(range(x.ndim))
    return torch.flip(x, dims=axis)


def concatenate(seq, axis=0, out=None):
    # XXX(nkoep): Why do we cast to float32 instead of float64 here?
    seq = [cast(t, float32) for t in seq]
    return torch.cat(seq, dim=axis, out=out)


def hstack(seq):
    return concatenate(seq, axis=1)


def vstack(seq):
    return concatenate(seq)


def array(val):
    if isinstance(val, list):
        if not any([isinstance(t, torch.Tensor) for t in val]):
            val = _np.copy(_np.array(val))
        elif any([not isinstance(t, torch.Tensor) for t in val]):
            for index, t in enumerate(val):
                if not isinstance(t, torch.Tensor):
                    val[index] = torch.tensor(t)
            val = stack(val)
        else:
            val = stack(val)

    if isinstance(val, (bool, int, float)):
        val = _np.array(val)
    if isinstance(val, _np.ndarray):
        if val.dtype == bool:
            val = torch.from_numpy(_np.array(val, dtype=_np.uint8))
        elif val.dtype == _np.float32 or val.dtype == _np.float64:
            val = torch.from_numpy(_np.array(val, dtype=_np.float64))
        else:
            val = torch.from_numpy(val)

    if not isinstance(val, torch.Tensor):
        val = torch.Tensor([val])
    if val.dtype == torch.float64:
        val = val.float()
    return val


def all(x, axis=None):
    if axis is None and torch.is_tensor(x):
        return x.bool().all()
    numpy_result = _np.array(_np.all(_np.array(x), axis=axis))
    return torch.from_numpy(numpy_result)


def get_slice(x, indices):
    """Return a slice of an array, following Numpy's style.

    Parameters
    ----------
    x : array-like, shape=[dimension]
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


def allclose(a, b, **kwargs):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
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
    return torch.allclose(a, b, **kwargs)


def arccosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1


def arcsinh(x):
    return torch.log(x + torch.sqrt(x * x + 1))


def arcosh(x):
    return torch.log(x + torch.sqrt(x * x - 1))


def shape(val):
    return val.shape


def dot(a, b):
    np_dot = _np.dot(a, b)
    return torch.from_numpy(_np.array(np_dot)).float()


def maximum(a, b):
    return torch.max(array(a), array(b))


def to_ndarray(x, to_ndim, axis=0):
    x = array(x)
    if x.dim() == to_ndim - 1:
        x = torch.unsqueeze(x, dim=axis)
    return x


def sqrt(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x).float()
    return torch.sqrt(x)


# TODO(nkoep): PyTorch exposes its own 'isclose' function, which is currently
#              undocumented for some reason, see
#                https://github.com/pytorch/pytorch/issues/35471
#              In the future, we may simply use that function instead.
def isclose(*args, **kwargs):
    return torch.from_numpy(_np.isclose(*args, **kwargs))


def sum(x, axis=None, keepdims=None, **kwargs):
    if axis is None:
        if keepdims is None:
            return torch.sum(x, **kwargs)
        return torch.sum(x, keepdim=keepdims, **kwargs)
    if keepdims is None:
        return torch.sum(x, dim=axis, **kwargs)
    return torch.sum(x, dim=axis, keepdim=keepdims, **kwargs)


def einsum(*args, **kwargs):
    einsum_str = args[0]
    input_tensors_list = args[1:]

    einsum_list = einsum_str.split('->')
    input_str = einsum_list[0]
    output_str = einsum_list[1]

    input_str_list = input_str.split(',')

    is_ellipsis = [input_str[:3] == '...' for input_str in input_str_list]
    all_ellipsis = bool(_np.prod(is_ellipsis))

    if all_ellipsis:
        if len(input_str_list) > 2:
            raise NotImplementedError(
                'Ellipsis support not implemented for >2 input tensors')
        tensor_a = input_tensors_list[0]
        tensor_b = input_tensors_list[1]
        n_tensor_a = tensor_a.shape[0]
        n_tensor_b = tensor_b.shape[0]

        if n_tensor_a != n_tensor_b:
            if n_tensor_a == 1:
                tensor_a = squeeze(tensor_a, axis=0)
                input_prefix_list = ['', 'r']
                output_prefix = 'r'
            elif n_tensor_b == 1:
                tensor_b = squeeze(tensor_b, axis=0)
                input_prefix_list = ['r', '']
                output_prefix = 'r'
            else:
                raise ValueError('Shape mismatch for einsum.')
        else:
            input_prefix_list = ['r', 'r']
            output_prefix = 'r'

        input_str_list = [
            input_str.replace('...', prefix) for input_str, prefix in zip(
                input_str_list, input_prefix_list)]
        output_str = output_str.replace('...', output_prefix)

        input_str = input_str_list[0] + ',' + input_str_list[1]
        einsum_str = input_str + '->' + output_str

        return torch.einsum(einsum_str, tensor_a, tensor_b, **kwargs)
    return torch.einsum(*args, **kwargs)


def T(x):
    return torch.t(x)


def transpose(x, axes=None):
    if axes:
        return x.permute(axes)
    if x.dim() == 1:
        return x
    return x.t()


def squeeze(x, axis=None):
    if axis is None:
        return torch.squeeze(x)
    return torch.squeeze(x, dim=axis)


def trace(*args, **kwargs):
    np_trace = _np.trace(*args, **kwargs)
    return torch.from_numpy(_np.array(np_trace)).float()


def arctanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def linspace(start, stop, num):
    return torch.linspace(start=start, end=stop, steps=num)


def equal(a, b, **kwargs):
    if a.dtype == torch.ByteTensor:
        a = cast(a, torch.uint8).float()
    if b.dtype == torch.ByteTensor:
        b = cast(b, torch.uint8).float()
    return torch.eq(a, b, **kwargs)


def cross(x, y):
    return torch.from_numpy(_np.cross(x, y))


def tril_indices(*args, **kwargs):
    return tuple(map(torch.from_numpy, _np.tril_indices(*args, **kwargs)))


def triu_indices(*args, **kwargs):
    return tuple(map(torch.from_numpy, _np.triu_indices(*args, **kwargs)))


def tile(x, y):
    # TODO(johmathe): Native tile implementation
    return array(_np.tile(x, y))


def expand_dims(x, axis=0):
    return torch.unsqueeze(x, dim=axis)


def ndim(x):
    return x.dim()


def hsplit(x, indices_or_section):
    if isinstance(indices_or_section, int):
        indices_or_section = indices_or_section // x.shape[1]
    return torch.split(x, indices_or_section, dim=1)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


def set_diag(x, new_diag):
    """Set the diagonal along the last two axis.

    Parameters
    ----------
    x : array-like, shape=[dimension]
        Initial array.
    new_diag : array-like, shape=[dimension[-2]]
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


def prod(x, axis=None):
    if axis is None:
        return torch.prod(x)
    return torch.prod(x, dim=axis)


def mean(x, axis=None):
    if axis is None:
        return torch.mean(x)
    return torch.mean(x, dim=axis)


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
    range_n = arange(cast(array(n), int32))
    i_float = cast(array(i), int32)
    mask_i = equal(range_n, i_float)
    mask_i_float = cast(mask_i, float32)
    return mask_i_float


def assignment(x, values, indices, axis=0):
    """Assign values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dimension]
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
    x_new : array-like, shape=[dimension]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)
    single_index = not isinstance(indices, list)
    if single_index:
        indices = [indices]
    if not isinstance(values, list):
        values = [values] * len(indices)
    for (nb_index, index) in enumerate(indices):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) < len(shape(x)):
            for n_axis in range(shape(x)[axis]):
                extended_index = index[:axis] + (n_axis,) + index[axis:]
                x_new[extended_index] = values[nb_index]
        else:
            x_new[index] = values[nb_index]
    return x_new


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dimension]
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
    x_new : array-like, shape=[dimension]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)
    single_index = not isinstance(indices, list)
    if single_index:
        indices = [indices]
    if not isinstance(values, list):
        values = [values] * len(indices)
    for (nb_index, index) in enumerate(indices):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) < len(shape(x)):
            for n_axis in range(shape(x)[axis]):
                extended_index = index[:axis] + (n_axis,) + index[axis:]
                x_new[extended_index] += values[nb_index]
        else:
            x_new[index] += values[nb_index]
    return x_new


def copy(x):
    return x.clone()


def cumsum(x, axis=None):
    if axis is None:
        return x.flatten().cumsum(dim=0)
    return torch.cumsum(x, dim=axis)


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
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices).t(),
        torch.FloatTensor(cast(data, float32)),
        torch.Size(target_shape)).to_dense()


def amax(*args, **kwargs):
    return torch.max(*args, **kwargs).values


def amin(*args, **kwargs):
    return torch.min(*args, **kwargs).values