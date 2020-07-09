"""Pytorch based computation backend."""

from functools import wraps

import numpy as _np
import torch
from torch import (  # NOQA
    abs,
    acos as arccos,
    arange,
    argmin,
    asin as arcsin,
    atan2 as arctan2,
    bool as t_bool,
    broadcast_tensors as broadcast_arrays,
    ceil,
    clamp as clip,
    cos,
    cosh,
    cross,
    div as divide,
    empty_like,
    eq,
    erf,
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
    log,
    lt as less,
    matmul,
    max as amax,
    mean,
    meshgrid,
    min as amin,
    nonzero,
    ones,
    ones_like,
    polygamma,
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
    uint8,
    zeros,
    zeros_like
)

from . import autograd # NOQA
from . import linalg  # NOQA
from . import random  # NOQA


DTYPES = {
    int32: 0,
    int64: 1,
    float32: 2,
    float64: 3}


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


searchsorted = _raise_not_implemented_error


def _box_scalar(function):
    @wraps(function)
    def wrapper(x):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return function(x)
    return wrapper


ceil = _box_scalar(ceil)
cos = _box_scalar(cos)
cosh = _box_scalar(cosh)
exp = _box_scalar(exp)
log = _box_scalar(log)
sin = _box_scalar(sin)
sinh = _box_scalar(sinh)


def to_numpy(x):
    return x.numpy()


def argmax(a, **kwargs):
    if a.dtype == torch.bool:
        return torch.as_tensor(_np.argmax(a.data.numpy(), **kwargs))
    return torch.argmax(a, **kwargs)


def convert_to_wider_dtype(tensor_list):
    dtype_list = [DTYPES[x.dtype] for x in tensor_list]
    wider_dtype_index = max(dtype_list)

    wider_dtype = list(DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return tensor_list


def less_equal(x, y, **kwargs):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    return torch.le(x, y, **kwargs)


def empty(shape, dtype=float64):
    return torch.empty(*shape, dtype=dtype)


def split(x, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, int):
        indices_or_sections = x.shape[axis] // indices_or_sections
        return torch.split(x, indices_or_sections, dim=axis)
    indices_or_sections = _np.array(indices_or_sections)
    intervals_length = indices_or_sections[1:] - indices_or_sections[:-1]
    last_interval_length = x.shape[axis] - indices_or_sections[-1]
    if last_interval_length > 0:
        intervals_length = _np.append(intervals_length, last_interval_length)
    intervals_length = _np.insert(intervals_length, 0, indices_or_sections[0])
    return torch.split(x, tuple(intervals_length), dim=axis)


def logical_or(x, y):
    return x or y


def logical_and(x, y):
    if torch.is_tensor(x):
        return x & y
    return x and y


def any(x, axis=None):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if axis is None:
        return torch.any(x)
    if isinstance(axis, int):
        return torch.any(x.bool(), axis)
    if len(axis) == 1:
        return torch.any(x, *axis)
    axis = list(axis)
    for i_axis, one_axis in enumerate(axis):
        if one_axis < 0:
            axis[i_axis] = ndim(x) + one_axis
    new_axis = tuple(k - 1 if k >= 0 else k for k in axis[1:])
    return any(torch.any(x.bool(), axis[0]), new_axis)


def cast(x, dtype):
    if torch.is_tensor(x):
        return x.to(dtype)
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


def _get_largest_dtype(seq):
    dtype_dict = {0: t_bool,
                  1: uint8,
                  2: int32,
                  3: int64,
                  4: float32,
                  5: float64}
    reverse_dict = {dtype_dict[key]: key for key in dtype_dict}
    dtype_code_set = {reverse_dict[t.dtype] for t in seq}
    return dtype_dict[max(dtype_code_set)]


def array(val, dtype=None):
    if isinstance(val, (list, tuple)):
        if isinstance(val[0], (list, tuple)):
            aux_list = [array(t, dtype) for t in val]
            if dtype is None:
                local_dtype = _get_largest_dtype(aux_list)
                aux_list = [cast(t, local_dtype) for t in aux_list]
            return stack(aux_list)
        if not any([isinstance(t, torch.Tensor) for t in val]):
            val = _np.copy(_np.array(val))
        elif any([not isinstance(t, torch.Tensor) for t in val]):
            tensor_members = [t for t in val if torch.is_tensor(t)]
            local_dtype = _get_largest_dtype(tensor_members)
            for index, t in enumerate(val):
                if torch.is_tensor(t) and t.dtype != local_dtype:
                    cast(t, local_dtype)
                elif torch.is_tensor(t):
                    val[index] = cast(t, dtype=local_dtype)
                else:
                    val[index] = torch.tensor(t, dtype=local_dtype)
            val = stack(val)
        else:
            val = stack(val)

    if isinstance(val, (bool, int, float)):
        val = _np.array(val)

    if isinstance(val, _np.ndarray):
        val = torch.from_numpy(val)

    if not isinstance(val, torch.Tensor):
        val = torch.Tensor([val])

    if dtype is not None:
        if val.dtype != dtype:
            val = cast(val, dtype)
    elif val.dtype == torch.float64:
        val = val.float()
    return val


def all(x, axis=None):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if axis is None:
        return x.bool().all()
    if isinstance(axis, int):
        return torch.all(x.bool(), axis)
    if len(axis) == 1:
        return torch.all(x, *axis)
    axis = list(axis)
    for i_axis, one_axis in enumerate(axis):
        if one_axis < 0:
            axis[i_axis] = ndim(x) + one_axis
    new_axis = tuple(k - 1 if k >= 0 else k for k in axis[1:])
    return all(torch.all(x.bool(), axis[0]), new_axis)


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
    return einsum('...i,...i->...', a, b)


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


def isclose(x, y, rtol=1e-5, atol=1e-8):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    return torch.isclose(x, y, atol=atol, rtol=rtol)


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

    input_tensors_list = convert_to_wider_dtype(
        input_tensors_list)

    if len(input_tensors_list) == 1:
        return torch.einsum(einsum_str, input_tensors_list)

    einsum_list = einsum_str.split('->')
    input_str = einsum_list[0]
    if len(einsum_list) > 1:
        output_str = einsum_list[1]

    input_str_list = input_str.split(',')

    is_ellipsis = [input_str[:3] == '...' for input_str in input_str_list]
    all_ellipsis = bool(_np.prod(is_ellipsis))

    if all_ellipsis:
        ndims = [len(input_str[3:]) for input_str in input_str_list]

        if len(input_str_list) > 2:
            raise NotImplementedError(
                'Ellipsis support not implemented for >2 input tensors')

        tensor_a = input_tensors_list[0]
        tensor_b = input_tensors_list[1]
        initial_ndim_a = tensor_a.ndim
        initial_ndim_b = tensor_b.ndim
        tensor_a = to_ndarray(tensor_a, to_ndim=ndims[0] + 1)
        tensor_b = to_ndarray(tensor_b, to_ndim=ndims[1] + 1)

        n_tensor_a = tensor_a.shape[0]
        n_tensor_b = tensor_b.shape[0]

        cond = (
            n_tensor_a == n_tensor_b == 1
            and initial_ndim_a != tensor_a.ndim
            and initial_ndim_b != tensor_b.ndim)

        if cond:
            tensor_a = squeeze(tensor_a, axis=0)
            tensor_b = squeeze(tensor_b, axis=0)
            input_prefix_list = ['', '']
            output_prefix = ''
        elif n_tensor_a != n_tensor_b:
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

        input_str = input_str_list[0] + ',' + input_str_list[1]

        einsum_str = input_str
        if len(einsum_list) > 1:
            output_str = output_str.replace('...', output_prefix)
            einsum_str = input_str + '->' + output_str

        result = torch.einsum(einsum_str, tensor_a, tensor_b, **kwargs)

        return result

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


def trace(x, axis1=0, axis2=1):
    min_axis = min(axis1, axis2)
    max_axis = max(axis1, axis2)
    if min_axis == 1 and max_axis == 2:
        return torch.einsum('...ii', x)
    if min_axis == -2 and max_axis == -1:
        return torch.einsum('...ii', x)
    if min_axis == 0 and max_axis == 1:
        return torch.einsum('ii...', x)
    if min_axis == 0 and max_axis == 2:
        return torch.einsum('i...i', x)
    raise NotImplementedError()


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


def tril_indices(*args, **kwargs):
    return tuple(map(torch.from_numpy, _np.tril_indices(*args, **kwargs)))


def triu_indices(*args, **kwargs):
    return tuple(map(torch.from_numpy, _np.triu_indices(*args, **kwargs)))


def tile(x, y):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.repeat(y)


def expand_dims(x, axis=0):
    return torch.unsqueeze(x, dim=axis)


def ndim(x):
    return x.dim()


def hsplit(x, indices_or_section):
    if isinstance(indices_or_section, int):
        indices_or_section = x.shape[1] // indices_or_section
    return torch.split(x, indices_or_section, dim=1)


def diagonal(x, offset=0, axis1=0, axis2=1):
    return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)


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


def prod(x, axis=None):
    if axis is None:
        return torch.prod(x)
    return torch.prod(x, dim=axis)


def where(condition, x=None, y=None):
    if x is None and y is None:
        return torch.where(condition)
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    y = cast(y, x.dtype)
    return torch.where(condition, x, y)


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


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return _is_boolean(x[0])
    if torch.is_tensor(x):
        return x.dtype in [torch.bool, torch.uint8]
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if torch.is_tensor(x):
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

    use_vectorization = hasattr(indices, '__len__') and len(indices) < ndim(x)
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
            raise ValueError('Either one value or as many values as indices')
        x_new[indices] = values
    else:
        indices = tuple(
            list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
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
    use_vectorization = hasattr(indices, '__len__') and len(indices) < ndim(x)
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
            raise ValueError('Either one value or as many values as indices')
        x_new[indices] += values
    else:
        indices = tuple(
            list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] += values
    return x_new


def copy(x):
    return x.clone()


def cumsum(x, axis=None):
    if not torch.is_tensor(x):
        x = array(x)
    if axis is None:
        return x.flatten().cumsum(dim=0)
    return torch.cumsum(x, dim=axis)


def cumprod(x, axis=None):
    if axis is None:
        return x.flatten().cumprod(dim=0)
    return torch.cumprod(x, dim=axis)


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


def vectorize(x, pyfunc, multiple_args=False, **kwargs):
    if multiple_args:
        return stack(list(map(lambda y: pyfunc(*y), zip(*x))))
    return stack(list(map(pyfunc, x)))
