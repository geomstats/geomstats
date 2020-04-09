"""Tensorflow based computation backend."""

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
    reduce_sum as sum,
    reshape,
    searchsorted,
    shape,
    sign,
    sin,
    sinh,
    split,
    sqrt,
    squeeze,
    stack,
    tan,
    tanh,
    tile,
    where,
    zeros,
    zeros_like
)



from . import linalg  # NOQA
from . import random  # NOQA

arctanh = tf.math.atanh
ceil = tf.math.ceil
cross = tf.linalg.cross
log = tf.math.log
matmul = tf.linalg.matmul
mod = tf.math.mod
real = tf.math.real
std = tf.math.reduce_std


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


# TODO(nkoep): The 'repeat' function was added in TF 2.1. Backport the
#              implementation from tensorflow/python/ops/array_ops.py.
repeat = _raise_not_implemented_error


def array(x, dtype=None):
    return tf.convert_to_tensor(x, dtype=dtype)


# TODO(nkoep): Handle the optional axis arguments.
def trace(a, axis1=0, axis2=1):
    return tf.linalg.trace(a)


# TODO(nkoep): Handle the optional axis arguments.
def diagonal(a, axis1=0, axis2=1):
    return tf.linalg.diag_part(a)


def ndim(x):
    return tf.convert_to_tensor(x).ndim


def to_ndarray(x, to_ndim, axis=0):
    if ndim(x) == to_ndim - 1:
        x = tf.expand_dims(x, axis=axis)
    return x


def empty(shape, dtype=float64):
    assert isinstance(dtype, tf.DType)
    np_dtype = dtype.as_numpy_dtype
    return tf.convert_to_tensor(_np.empty(shape, dtype=np_dtype))


def empty_like(prototype, dtype=None):
    shape = tf.shape(prototype)
    if dtype is None:
        dtype = prototype.dtype
    return empty(shape, dtype=dtype)


def flip(m, axis=None):
    assert isinstance(m, tf.Tensor)
    if axis is None:
        axis = range(m.ndim)
    elif not hasattr(axis, '__iter__'):
        axis = (axis,)
    return tf.reverse(m, axis=axis)


def any(x, axis=None):
    return tf.math.reduce_any(tf.cast(x, bool), axis=axis)


def get_mask_i_float(i, n):
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

    for (nb_index, index) in enumerate(indices):
        if not isinstance(index, tuple):
            indices[nb_index] = (index,)

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
    x: array-like, shape=[dimension]
        Initial array which will be copied.
    n_samples: int
        Number of copies of the array to create.
    axis: int, optional
        Dimension of the new array along which the copies of x are made.

    Returns
    -------
    tiled_array: array, shape=[dimension[:axis], n_samples, dimension[axis:]]
        Copies of x stacked along dimension axis
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


def _assignment_single_value_by_sum(x, value, indices, axis=0):
    """Add a value at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dimension]
        Initial array.
    value: float
        Value to be added.
    indices: {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), value is
        assigned to each copy along axis.
    axis: int, optional
        Axis along which value is assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dimension]
        Copy of x where value was added at all indices (and possibly along
        an axis).
    """
    single_index = not isinstance(indices, list)
    if single_index:
        indices = [indices]
    if isinstance(indices[0], tuple):
        use_vectorization = (len(indices[0]) < ndim(x))
    else:
        use_vectorization = ndim(x) > 1

    if use_vectorization:
        n_samples = shape(x).numpy()[0]
        mask = _vectorized_mask_from_indices(
            n_samples, indices, shape(x).numpy()[1:], axis, x.dtype)
    else:
        mask = _mask_from_indices(indices, shape(x), x.dtype)
    x_new = x + value * mask
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
        Copy of x as the sum of x and the values at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    if not isinstance(values, list):
        return _assignment_single_value_by_sum(x, values, indices, axis)

    if not isinstance(indices, list):
        indices = [indices]

    if len(values) != len(indices):
        raise ValueError('Either one value or as many values as indices')

    for (nb_index, index) in enumerate(indices):
        x = _assignment_single_value_by_sum(
            x, values[nb_index], index, axis)
    return x


def _assignment_single_value(x, value, indices, axis=0):
    """Assign a value at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dimension]
        Initial array.
    value: float
        Value to be added.
    indices: {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), value is
        assigned to each copy along axis.
    axis: int, optional
        Axis along which value is assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dimension]
        Copy of x where value was assigned at all indices (and possibly
        along an axis).
    """
    single_index = not isinstance(indices, list)
    if single_index:
        indices = [indices]
    if isinstance(indices[0], tuple):
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
    x_new = x + -x * mask + value * mask
    return x_new


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
    if not isinstance(values, list):
        return _assignment_single_value(x, values, indices, axis)

    if not isinstance(indices, list):
        indices = [indices]

    if len(values) != len(indices):
        raise ValueError('Either one value or as many values as indices')

    for (nb_index, index) in enumerate(indices):
        x = _assignment_single_value(x, values[nb_index], index, axis)
    return x


def array_from_sparse(indices, data, target_shape):
    return tf.sparse.to_dense(tf.sparse.reorder(
        tf.SparseTensor(indices, data, target_shape)))


def vectorize(x, pyfunc, multiple_args=False, dtype=None, **kwargs):
    if multiple_args:
        return tf.map_fn(lambda x: pyfunc(*x), elems=x, dtype=dtype)
    return tf.map_fn(pyfunc, elems=x, dtype=dtype)


def hsplit(x, n_splits):
    return tf.split(x, num_or_size_splits=n_splits, axis=1)


def flatten(x):
    """Collapses the tensor into 1-D.

    Following https://www.tensorflow.org/api_docs/python/tf/reshape"""
    return tf.reshape(x, [-1])


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


def dot(x, y):
    return tf.tensordot(x, y, axes=1)


def isclose(x, y, rtol=1e-05, atol=1e-08):
    rhs = tf.constant(atol) + tf.constant(rtol) * tf.abs(y)
    return tf.less_equal(tf.abs(tf.subtract(x, y)), rhs)


def allclose(x, y, rtol=1e-05, atol=1e-08):
    return tf.reduce_all(isclose(x, y, rtol=rtol, atol=atol))


def eye(n, m=None):
    if m is None:
        m = n
    return tf.eye(num_rows=n, num_columns=m)


def einsum(equation, *inputs, **kwargs):
    einsum_str = equation
    input_tensors_list = inputs

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

        return tf.einsum(einsum_str, tensor_a, tensor_b, **kwargs)

    return tf.einsum(equation, *inputs, **kwargs)


def transpose(x, axes=None):
    return tf.transpose(x, perm=axes)


def all(x, axis=None):
    return tf.math.reduce_all(tf.cast(x, bool), axis=axis)


def cumsum(a, axis=None):
    if axis is None:
        return tf.math.cumsum(flatten(a), axis=0)
    return tf.math.cumsum(a, axis=axis)


def tril(m, k=0):
    if k != 0:
        raise NotImplementedError("Only k=0 supported so far")
    return tf.linalg.band_part(m, -1, 0)


def tril_indices(*args, **kwargs):
    return tuple(map(tf.convert_to_tensor, _np.tril_indices(*args, **kwargs)))


def triu_indices(*args, **kwargs):
    return tuple(map(tf.convert_to_tensor, _np.triu_indices(*args, **kwargs)))
