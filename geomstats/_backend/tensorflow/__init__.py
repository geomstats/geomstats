"""Tensorflow based computation backend."""

from collections import Counter
from itertools import product

import numpy as _np
import tensorflow as tf
from tensorflow import (  # NOQA
    abs,
    acos as arccos,
    acosh as arccosh,
    argmax,
    argmin,
    asin as arcsin,
    atan2 as arctan2,
    clip_by_value as clip,
    concat as concatenate,
    cos,
    cosh,
    divide,
    equal,
    exp,
    expand_dims,
    float32,
    float64,
    floor,
    gather,
    greater,
    int32,
    int64,
    less,
    less_equal,
    linspace,
    logical_and,
    logical_or,
    maximum,
    meshgrid,
    ones,
    ones_like,
    range as arange,
    reduce_max as amax,
    reduce_mean as mean,
    reduce_min as amin,
    reshape,
    searchsorted,
    shape,
    sign,
    sin,
    sinh,
    sqrt,
    squeeze,
    stack,
    tan,
    tanh,
    tile,
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


arctanh = tf.math.atanh
ceil = tf.math.ceil
cross = tf.linalg.cross
erf = tf.math.erf
isnan = tf.math.is_nan
log = tf.math.log
mod = tf.math.mod
polygamma = tf.math.polygamma
power = tf.math.pow
real = tf.math.real
set_diag = tf.linalg.set_diag
std = tf.math.reduce_std


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


def to_numpy(x):
    return x.numpy()


def convert_to_wider_dtype(tensor_list):
    dtype_list = [DTYPES[x.dtype] for x in tensor_list]
    wider_dtype_index = max(dtype_list)

    wider_dtype = list(DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return tensor_list


def repeat(a, repeats, axis=None):
    return tf.repeat(input=a, repeats=repeats, axis=axis)


def array(x, dtype=None):
    return tf.convert_to_tensor(x, dtype=dtype)


def trace(x, axis1=0, axis2=1):
    min_axis = min(axis1, axis2)
    max_axis = max(axis1, axis2)
    if min_axis == 1 and max_axis == 2:
        return tf.einsum('...ii', x)
    if min_axis == -2 and max_axis == -1:
        return tf.einsum('...ii', x)
    if min_axis == 0 and max_axis == 1:
        return tf.einsum('ii...', x)
    if min_axis == 0 and max_axis == 2:
        return tf.einsum('i...i', x)
    raise NotImplementedError()


# TODO (nkoep): Handle the optional axis arguments.
def diagonal(a, axis1=0, axis2=1):
    return tf.linalg.diag_part(a)


def ndim(x):
    return tf.convert_to_tensor(x).ndim


def to_ndarray(x, to_ndim, axis=0):
    if ndim(x) == to_ndim - 1:
        x = tf.expand_dims(x, axis=axis)
    return x


def empty(shape, dtype=float64):
    if not isinstance(dtype, tf.DType):
        raise ValueError('dtype must be one of Tensorflow\'s types')
    np_dtype = dtype.as_numpy_dtype
    return tf.convert_to_tensor(_np.empty(shape, dtype=np_dtype))


def empty_like(prototype, dtype=None):
    initial_shape = tf.shape(prototype)
    if dtype is None:
        dtype = prototype.dtype
    return empty(initial_shape, dtype=dtype)


def flip(m, axis=None):
    if not isinstance(m, tf.Tensor):
        raise ValueError('m must be a Tensorflow tensor')
    if axis is None:
        axis = range(m.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = (axis,)
    return tf.reverse(m, axis=axis)


def any(x, axis=None):
    return tf.math.reduce_any(tf.cast(x, bool), axis=axis)


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return isinstance(x[0], bool)
    if tf.is_tensor(x):
        return x.dtype == bool
    return False


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


def _mask_from_indices(indices, mask_shape, dtype=float32):
    """Create a binary mask.

    Parameters
    ----------
    indices: tuple
        Single index or tuple of indices where ones will be.
    mask_shape: tuple
        Shape of the mask.
    dtype: dtype
        Type of the mask.

    Returns
    -------
    tf_mask : array, shape=[mask_shape]
    """
    np_mask = _np.zeros(mask_shape)

    for i_index, index in enumerate(indices):
        if not isinstance(index, tuple):
            if hasattr(index, '__iter__'):
                indices[i_index] = tuple(index)
            else:
                indices[i_index] = (index,)
    for index in indices:
        if len(index) != len(mask_shape):
            raise ValueError('Indices must have the same size as shape')

    for index in indices:
        np_mask[index] = 1
    tf_mask = array(np_mask, dtype=dtype)
    return tf_mask


def _duplicate_array(x, n_samples, axis=0):
    """Stack copies of an array along an additional dimension.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array which will be copied.
    n_samples: int
        Number of copies of the array to create.
    axis: int, optional
        Dimension of the new array along which the copies of x are made.

    Returns
    -------
    tiled_array: array, shape=[dim[:axis], n_samples, dim[axis:]]
        Copies of x stacked along dim axis
    """
    multiples = _np.ones(ndim(x) + 1, dtype=_np.int32)
    multiples[axis] = n_samples
    return tile(to_ndarray(x, ndim(x) + 1, axis), multiples)


def _vectorized_mask_from_indices(
        n_samples=1, indices=None, mask_shape=None, axis=0, dtype=float32):
    """Create a vectorized binary mask.

    Parameters
    ----------
    n_samples: int
        Number of copies of the mask along the additional dimension.
    indices : {tuple, list(tuple)}
        Single tuple, or list of tuples of indices where ones will be.
    mask_shape : tuple
        Shape of the mask.
    axis: int
        Axis along which the mask is vectorized.
    dtype: dtype
        Type of the returned array.

    Returns
    -------
    tf_mask : array, shape=[mask_shape[:axis], n_samples, mask_shape[axis:]]
    """
    mask = _mask_from_indices(indices, mask_shape, dtype)
    return _duplicate_array(mask, n_samples, axis=axis)


def _assignment_single_value(x, value, indices, mode='replace', axis=0):
    """Assign a value at given indices of an array.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    value : float
        Value to be added.
    indices : {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), value is
        assigned to each copy along axis.
    mode : string, optional
        Whether the assignment is done by replacing the old value,
        or by adding to it. Possible values are 'replace' and 'sum'
    axis : int, optional
        Axis along which value is assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x where value was assigned at all indices (and possibly
        along an axis).
    """
    single_index = not isinstance(indices, list)
    if tf.is_tensor(indices):
        single_index = ndim(indices) <= 1 and sum(indices.shape) <= ndim(x)
    if single_index:
        indices = [indices]

    if isinstance(indices[0], tuple):
        use_vectorization = (len(indices[0]) < ndim(x))
    elif tf.is_tensor(indices[0]) and ndim(indices[0]) >= 1:
        use_vectorization = (len(indices[0]) < ndim(x))
    else:
        use_vectorization = ndim(x) > 1

    if use_vectorization:
        full_shape = shape(x).numpy()
        n_samples = full_shape[axis]
        tile_shape = list(full_shape[:axis]) + list(full_shape[axis + 1:])
        mask = _vectorized_mask_from_indices(
            n_samples, indices, tile_shape, axis, x.dtype)
    else:
        mask = _mask_from_indices(indices, shape(x), x.dtype)
    if mode == 'replace':
        return x + -x * mask + value * mask
    if mode == 'sum':
        return x + value * mask
    raise ValueError('mode must be one of \'replace\' or \'sum\'')


def _assignment(x, values, indices, mode, axis):
    if _is_boolean(indices):
        if ndim(array(indices)) > 1:
            indices_tensor = tf.where(indices)
            indices = [tuple(ind) for ind in indices_tensor]
        else:
            indices_from_booleans = [
                index for index, val in enumerate(indices) if val]
            indices_along_dims = [range(dim) for dim in shape(x)]
            indices_along_dims[axis] = indices_from_booleans
            indices = list(product(*indices_along_dims))
    if tf.rank(values) == 0:
        return _assignment_single_value(x, values, indices, mode, axis)
    values = cast(flatten(array(values)), x.dtype)

    single_index = not isinstance(indices, list)
    if tf.is_tensor(indices):
        single_index = ndim(indices) <= 1 and sum(indices.shape) <= ndim(x)
    if single_index:
        if len(values) > 1:
            indices = [tuple(list(indices[:axis]) + [i] + list(indices[axis:]))
                       for i in range(x.shape[axis])]
        else:
            indices = [indices]

    if len(values) != len(indices):
        raise ValueError('Either one value or as many values as indices')

    for i_index, index in enumerate(indices):
        x = _assignment_single_value(x, values[i_index], index, mode, axis)
    return x


def assignment(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {
    int, tuple(int), array-like({int, tuple, boolean})
        Single index or array of indices where values are assigned.
        If the length of the tuples is shorter than ndim(x) by one, values are
        assigned to each copy along axis.
        If indices is a list of booleans and ndim(x) > 1, values are assigned
        across all dimensions.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x as the sum of x and the values at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a single index is provided, and len(indices) == ndim(x) - 1, then values
    are assigned along axis.

    Examples
    --------
    Most examples translate as
    assignment(x, indices, values) <=> x[indices] = values
    Some special cases are given by vectorisation.
    (Beware that copies are always returned).
    if ndim(x) == 3, assignment(x, 1, (1, 0), 1) <=> x[1, :, 0] = 1
    if ndim(x) == 2, assignment(x, [1, 2], [(0, 1), (2, 3)]) <=>
                        x[((0, 2), (1, 3))] = [1, 2]
    """
    return _assignment(x, values, indices, 'replace', axis)


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {
    int, tuple(int), array-like({int, tuple, boolean})
        Single index or array of indices where values are assigned.
        If the length of the tuples is shorter than ndim(x) by one, values are
        assigned to each copy along axis.
        If indices is a list of booleans and ndim(x) > 1, values are assigned
        across all dimensions.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x as the sum of x and the values at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a single index is provided, and len(indices) == ndim(x) - 1, then values
    are assigned along axis.

    Examples
    --------
    Most examples translate as
    assignment_by_sum(x, indices, values) <=> x[indices] = x[indices] + values
    Some special cases are given by vectorisation.
    (Beware that copies are always returned).
    if ndim(x) == 3, assignment_by_sum(x, 1, (1, 0), 1) <=> x[1, :, 0] += 1
    if ndim(x) == 2, assignment_by_sum(x, [1, 2], [(0, 1), (2, 3)]) <=>
                        x[((0, 2), (1, 3))] += [1, 2]
    """
    return _assignment(x, values, indices, 'sum', axis)


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
    return tf.sparse.to_dense(tf.sparse.reorder(
        tf.SparseTensor(indices, data, target_shape)))


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
    >>> a = tf.reshape(tf.convert_to_tensor(range(30)), (3,10))
    >>> get_slice(a, ((0, 2), (8, 9)))
    <tf.Tensor: id=41, shape=(2,), dtype=int32, numpy=array([ 8, 29])>
    """
    if hasattr(indices, 'shape'):
        if indices.shape.rank == 0:
            return x[indices]

        if tf.is_tensor(indices) and indices.shape[-1] == 1:
            return tf.gather_nd(x, indices)

    return tf.gather_nd(x, list(zip(*indices)))


def vectorize(x, pyfunc, multiple_args=False, dtype=None, **kwargs):
    if multiple_args:
        return tf.map_fn(lambda y: pyfunc(*y), elems=x, dtype=dtype)
    return tf.map_fn(pyfunc, elems=x, dtype=dtype)


def split(x, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, int):
        return tf.split(x, indices_or_sections, dim=axis)
    indices_or_sections = _np.array(indices_or_sections)
    intervals_length = indices_or_sections[1:] - indices_or_sections[:-1]
    last_interval_length = x.shape[axis] - indices_or_sections[-1]
    if last_interval_length > 0:
        intervals_length = _np.append(intervals_length, last_interval_length)
    intervals_length = _np.insert(intervals_length, 0, indices_or_sections[0])
    return tf.split(x, num_or_size_splits=tuple(intervals_length), axis=axis)


def hsplit(x, n_splits):
    return tf.split(x, num_or_size_splits=n_splits, axis=1)


def flatten(x):
    """Collapse the tensor into 1-D.

    Following https://www.tensorflow.org/api_docs/python/tf/reshape
    """
    return tf.reshape(x, [-1])


def matmul(a, b):
    """Matrix-matrix or matrix-vector product of two tensors.

    This wraps both mathvec and matmul into a single function, to mimic the
    behavior of torch's and numpy's versions of matmul
    """
    if ndim(b) < ndim(a):
        if ndim(b) == 1 or b.shape[-2] != a.shape[-1]:
            return tf.linalg.matvec(a, b)
    return tf.linalg.matmul(a, b)


def outer(x, y):
    return tf.einsum('i,j->ij', x, y)


def copy(x):
    return tf.Variable(x)


def hstack(x):
    return tf.concat(x, axis=1)


def vstack(x):
    return tf.concat(x, axis=0)


def cast(x, dtype):
    return tf.cast(x, dtype)


def broadcast_arrays(x, y, **kwargs):
    tensors = [x, y]
    shapes = [t.get_shape().as_list() for t in tensors]
    max_rank = max(len(s) for s in shapes)

    for index, value in enumerate(shapes):
        shape = value
        if len(shape) == max_rank:
            continue

        tensor = tensors[index]
        for _ in range(max_rank - len(shape)):
            shape.insert(0, 1)
            tensor = tf.expand_dims(tensor, axis=0)
        tensors[index] = tensor

    broadcast_shape = []
    for index in range(max_rank):
        dimensions = [s[index] for s in shapes]
        repeats = Counter(dimensions)
        if len(repeats) > 2 or (len(repeats) == 2 and
                                1 not in list(repeats.keys())):
            raise ValueError('operands could not be '
                             'broadcast together with shapes', shapes)
        broadcast_shape.append(max(repeats.keys()))

    for axis, dimension in enumerate(broadcast_shape):
        tensors = [tf.concat([t] * dimension, axis=axis)
                   if t.get_shape()[axis] == 1 else t for t in tensors]

    return tensors


def dot(x, y):
    return tf.tensordot(x, y, axes=1)


def isclose(x, y, rtol=1e-05, atol=1e-08):
    if not tf.is_tensor(x):
        x = tf.constant(x)
    if not tf.is_tensor(y):
        y = tf.constant(y)
    x, y = convert_to_wider_dtype([x, y])
    dtype = x.dtype

    rhs = (
        tf.constant(atol, dtype=dtype)
        + tf.constant(rtol, dtype=dtype) * tf.abs(y))
    return tf.less_equal(tf.abs(tf.subtract(x, y)), rhs)


def allclose(x, y, rtol=1e-05, atol=1e-08):
    return tf.reduce_all(isclose(x, y, rtol=rtol, atol=atol))


def eye(n, m=None):
    if m is None:
        m = n
    return tf.eye(num_rows=n, num_columns=m)


def sum(x, axis=None, keepdims=False, name=None):
    if not tf.is_tensor(x):
        x = tf.convert_to_tensor(x)
    if x.dtype == bool:
        x = cast(x, int32)
    return tf.reduce_sum(x, axis, keepdims, name)


def einsum(equation, *inputs, **kwargs):
    einsum_str = equation
    input_tensors_list = inputs

    input_tensors_list = convert_to_wider_dtype(input_tensors_list)

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
        ndims = [len(input_str[3:]) for input_str in input_str_list]

        tensor_a = input_tensors_list[0]
        tensor_b = input_tensors_list[1]
        initial_ndim_a = tensor_a.ndim
        initial_ndim_b = tensor_b.ndim
        tensor_a = to_ndarray(tensor_a, to_ndim=ndims[0] + 1)
        tensor_b = to_ndarray(tensor_b, to_ndim=ndims[1] + 1)

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

        result = tf.einsum(einsum_str, tensor_a, tensor_b, **kwargs)

        cond = (
            n_tensor_a == n_tensor_b == 1
            and initial_ndim_a != tensor_a.ndim
            and initial_ndim_b != tensor_b.ndim)

        if cond:
            result = squeeze(result, axis=0)
        return result

    return tf.einsum(equation, *input_tensors_list, **kwargs)


def transpose(x, axes=None):
    return tf.transpose(x, perm=axes)


def all(x, axis=None):
    return tf.math.reduce_all(tf.cast(x, bool), axis=axis)


def cumsum(a, axis=None):
    if axis is None:
        return tf.math.cumsum(flatten(a), axis=0)
    return tf.math.cumsum(a, axis=axis)


def cumprod(a, axis=None):
    if axis is None:
        return tf.math.cumprod(flatten(a), axis=0)
    return tf.math.cumprod(a, axis=axis)


def tril(m, k=0):
    if k != 0:
        raise NotImplementedError("Only k=0 supported so far")
    return tf.linalg.band_part(m, -1, 0)


def tril_indices(*args, **kwargs):
    return tuple(map(tf.convert_to_tensor, _np.tril_indices(*args, **kwargs)))


def triu_indices(*args, **kwargs):
    return tuple(
        map(tf.convert_to_tensor, _np.triu_indices(*args, **kwargs)))


def where(condition, x=None, y=None):
    if x is None and y is None:
        return tf.where(condition)
    if not tf.is_tensor(x):
        x = tf.constant(x)
    if not tf.is_tensor(y):
        y = tf.constant(y)
    y = cast(y, x.dtype)
    return tf.where(condition, x, y)
