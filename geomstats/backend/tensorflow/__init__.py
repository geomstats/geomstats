"""Tensorflow based computation backend."""

import numpy as _np
import tensorflow as tf

from .common import array, ndim, to_ndarray  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA


int8 = tf.int8
int32 = tf.int32
int64 = tf.int64
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64


def while_loop(*args, **kwargs):
    return tf.while_loop(*args, **kwargs)


def logical_or(x, y):
    return tf.logical_or(x, y)


def logical_and(x, y):
    return tf.logical_and(x, y)


def any(x, axis=0):
    return tf.math.reduce_any(tf.cast(x, bool), axis=axis)


def get_mask_i_float(i, n):
    range_n = arange(n)
    i_float = cast(array([i]), int32)[0]
    mask_i = equal(range_n, i_float)
    mask_i_float = cast(mask_i, float32)
    return mask_i_float


def get_mask_float(indices, mask_shape):
    """Create a binary mask.

    Parameters
    ----------
    indices : tuple
        Single index or tuple of indices where ones will be.
    mask_shape : tuple
        Shape of the mask.

    Returns
    -------
    tf_mask : array
    """
    np_mask = _np.zeros(mask_shape)
    if ndim(array(indices)) <= 1 and ndim(np_mask) == 1:
        indices = list(indices)

    if ndim(array(indices)) == 1 and ndim(np_mask) > 1:
        if len(indices) != len(mask_shape):
            raise ValueError('The index must have the same size as shape')
        indices = [indices]

    else:
        for index in indices:
            if len(index) != len(mask_shape):
                raise ValueError('Indices must have the same size as shape')

    for index in indices:
        np_mask[index] = 1
    tf_mask = array(np_mask, dtype=float32)
    return tf_mask


def duplicate_array(x, n_samples, axis=0):
    """Stack copies of an array along an additional dimension.

    Parameters
    ----------
    x: array-like, shape=dimension
        Initial array which will be copied.
    n_samples: int
        Number of copies of the array to create.
    axis: int, optional
        Dimension of the new array along which the copies of x are made.

    Returns
    -------
    tiled_array: array, shape=[dimension[:axis], n_samples, dimension[aixs:]]
        Copies of x stacked along dimension axis
    """
    multiples = _np.ones(ndim(x) + 1, dtype=_np.int32)
    multiples[axis] = n_samples
    return tile(to_ndarray(x, ndim(x) + 1), multiples)


def get_vectorized_mask_float(
        n_samples=1, indices=None, mask_shape=None, axis=0):
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

    Returns
    -------
    tf_mask : array, shape=[mask_shape[:axis], n_samples, mask_shape[axis:]]
    """
    mask = get_mask_float(indices, mask_shape)
    return duplicate_array(mask, n_samples, axis=axis)


def assignment_single_value_by_sum(x, value, indices, axis=0):
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
    use_vectorization = (len(indices[0]) < ndim(x))

    if use_vectorization:
        n_samples = shape(x).numpy()[0]
        mask = get_vectorized_mask_float(
            n_samples, indices, shape(x).numpy()[1:], axis)
    else:
        mask = get_mask_float(indices, shape(x))
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
        return assignment_single_value_by_sum(x, values, indices, axis)

    else:
        if not isinstance(indices, list):
            indices = [indices]

        if len(values) != len(indices):
            raise ValueError('Either one value or as many values as indices')

        for (nb_index, index) in enumerate(indices):
            x = assignment_single_value_by_sum(
                x, values[nb_index], index, axis)
        return x


def assignment_single_value(x, value, indices, axis=0):
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
    use_vectorization = (len(indices[0]) < ndim(x))

    if use_vectorization:
        n_samples = shape(x).numpy()[0]
        mask = get_vectorized_mask_float(
            n_samples, indices, shape(x).numpy()[1:], axis)
    else:
        mask = get_mask_float(indices, shape(x))
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
        return assignment_single_value(x, values, indices, axis)

    else:
        if not isinstance(indices, list):
            indices = [indices]

        if len(values) != len(indices):
            raise ValueError('Either one value or as many values as indices')

        for (nb_index, index) in enumerate(indices):
            x = assignment_single_value(x, values[nb_index], index, axis)
        return x


def array_from_sparse(indices, data, target_shape):
    return tf.sparse.to_dense(tf.sparse.reorder(
        tf.SparseTensor(indices, data, target_shape)))


def gather(*args, **kwargs):
    return tf.gather(*args, **kwargs)


def where(*args, **kwargs):
    return tf.where(*args, **kwargs)


def vectorize(x, pyfunc, multiple_args=False, dtype=None, **kwargs):
    if multiple_args:
        return tf.map_fn(lambda x: pyfunc(*x), elems=x, dtype=dtype)
    return tf.map_fn(pyfunc, elems=x, dtype=dtype)


def sign(x):
    return tf.sign(x)


def hsplit(x, n_splits):
    return tf.split(x, num_or_size_splits=n_splits, axis=1)


def amax(x):
    return tf.reduce_max(x)


def real(x):
    return tf.math.real(x)


def cond(*args, **kwargs):
    return tf.cond(*args, **kwargs)


def reshape(*args, **kwargs):
    return tf.reshape(*args, **kwargs)


def flatten(x):
    """Collapses the tensor into 1-D.

    Following https://www.tensorflow.org/api_docs/python/tf/reshape"""
    return tf.reshape(x, [-1])


def arange(*args, **kwargs):
    return tf.range(*args, **kwargs)


def outer(x, y):
    return tf.einsum('i,j->ij', x, y)


def copy(x):
    return tf.Variable(x)


def linspace(start, stop, num):
    return tf.linspace(start, stop, num)


def mod(x, y):
    return tf.mod(x, y)


def boolean_mask(x, mask, name='boolean_mask', axis=None):
    return tf.boolean_mask(x, mask, name, axis)


def exp(x):
    return tf.exp(x)


def log(x):
    return tf.math.log(x)


def hstack(x):
    return tf.concat(x, axis=1)


def vstack(x):
    return tf.concat(x, axis=0)


def cast(x, dtype):
    return tf.cast(x, dtype)


def divide(x1, x2):
    return tf.divide(x1, x2)


def tile(x, reps):
    return tf.tile(x, reps)


def eval(x):
    if tf.executing_eagerly():
        return x
    return x.eval()


def abs(x):
    return tf.abs(x)


def zeros(x):
    return tf.zeros(x)


def ones(x):
    return tf.ones(x)


def sin(x):
    return tf.sin(x)


def cos(x):
    return tf.cos(x)


def cosh(x):
    return tf.cosh(x)


def sinh(x):
    return tf.sinh(x)


def tanh(x):
    return tf.tanh(x)


def arccosh(x):
    return tf.acosh(x)


def tan(x):
    return tf.tan(x)


def arcsin(x):
    return tf.asin(x)


def arccos(x):
    return tf.acos(x)


def shape(x):
    return tf.shape(x)


def dot(x, y):
    return tf.tensordot(x, y, axes=1)


def maximum(x, y):
    return tf.maximum(x, y)


def greater(x, y):
    return tf.greater(x, y)


def greater_equal(x, y):
    return tf.greater_equal(x, y)


def equal(x, y):
    return tf.equal(x, y)


def sqrt(x):
    return tf.sqrt(x)


def isclose(x, y, rtol=1e-05, atol=1e-08):
    rhs = tf.constant(atol) + tf.constant(rtol) * tf.abs(y)
    return tf.less_equal(tf.abs(tf.subtract(x, y)), rhs)


def allclose(x, y, rtol=1e-05, atol=1e-08):
    return tf.reduce_all(isclose(x, y, rtol=rtol, atol=atol))


def less(x, y):
    return tf.less(x, y)


def less_equal(x, y):
    return tf.less_equal(x, y)


def eye(n, m=None):
    if m is None:
        m = n
    n = cast(n, dtype=int32)
    m = cast(m, dtype=int32)
    return tf.eye(num_rows=n, num_columns=m)


def matmul(x, y):
    return tf.matmul(x, y)


def argmax(*args, **kwargs):
    return tf.argmax(*args, **kwargs)


def sum(*args, **kwargs):
    return tf.reduce_sum(*args, **kwargs)


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


def squeeze(x, **kwargs):
    return tf.squeeze(x, **kwargs)


def zeros_like(x):
    return tf.zeros_like(x)


def ones_like(x):
    return tf.ones_like(x)


def trace(x, **kwargs):
    return tf.linalg.trace(x)


def all(bool_tensor, axis=None, keepdims=False):
    bool_tensor = tf.cast(bool_tensor, tf.bool)
    all_true = tf.reduce_all(bool_tensor, axis, keepdims)
    return all_true


def concatenate(*args, **kwargs):
    return tf.concat(*args, **kwargs)


def asarray(x):
    return x


def expand_dims(x, axis=None):
    return tf.expand_dims(x, axis)


def clip(x, min_value, max_value):
    return tf.clip_by_value(x, min_value, max_value)


def floor(x):
    return tf.floor(x)


def diag(a):
    return tf.map_fn(
        lambda x: tf.linalg.tensor_diag(x),
        a)


def cross(a, b):
    return tf.cross(a, b)


def stack(*args, **kwargs):
    return tf.stack(*args, **kwargs)


def unstack(*args, **kwargs):
    return tf.unstack(*args, **kwargs)


def arctan2(*args, **kwargs):
    return tf.atan2(*args, **kwargs)


def diagonal(*args, **kwargs):
    return tf.linalg.diag_part(*args)


def mean(x, axis=None):
    return tf.reduce_mean(x, axis)


def argmin(*args, **kwargs):
    return tf.argmin(*args, **kwargs)


def cumprod(x, axis=0):
    if axis is None:
        raise NotImplementedError('cumprod is not defined where axis is None')
    else:
        return tf.math.cumprod(x, axis=axis)


def from_vector_to_diagonal_matrix(x):
    n = shape(x)[-1]
    identity = eye(n)
    diagonals = einsum('ki,ij->kij', x, identity)
    return diagonals


def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    return _np.triu_indices(n, k, m)
