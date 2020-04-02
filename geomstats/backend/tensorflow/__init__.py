"""Tensorflow based computation backend."""

import numpy as _np
import tensorflow as tf
from tensorflow import (  # NOQA
    abs,
    argmax,
    argmin,
    cond,
    cos,
    cosh,
    cross,
    einsum,
    equal,
    exp,
    floor,
    gather,
    greater,
    greater_equal,
    less,
    less_equal,
    logical_and,
    logical_or,
    maximum,
    mod,
    ones,
    ones_like,
    reshape,
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
    transpose,
    unstack,
    while_loop,
    zeros,
    zeros_like
)

from .common import array, ndim, to_ndarray  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA


int8 = tf.int8
int32 = tf.int32
int64 = tf.int64
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64


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
    multiples = _np.ones(ndim(x) + 1, dtype=_np.int32)
    multiples[axis] = n_samples
    return tile(to_ndarray(x, ndim(x) + 1), multiples)


def get_vectorized_mask_float(
        n_samples=1, indices=None, mask_shape=None, axis=0):
    """Create a vectorized binzary mask.

    Parameters
    ----------
    n_samples: int
        Number of copies of the mask along the additional dimension.
    indices : tuple, optional
        Single index or tuple of indices where ones will be.
    mask_shape : tuple, optional
        Shape of the mask.
    axis: int
        Axis along which the mask is vectorized.

    Returns
    -------
    tf_mask : array
    TODO(pchauchat): give shape of the output according to guideline
    """
    mask = get_mask_float(indices, mask_shape)
    return duplicate_array(mask, n_samples, axis=axis)


def assignment_single_value(x, value, indices, axis=0):
    single_index = ndim(array(indices)) == 0 or (ndim(array(indices)) <= 1 and ndim(x) > 1)
    if single_index:
        indices = [indices]
    use_vectorization = (len(indices[0]) < ndim(x))

    if use_vectorization:
        n_samples = shape(x).numpy()[0]
        mask = get_vectorized_mask_float(
            n_samples, indices, shape(x).numpy()[1:], axis)
    else:
        mask = get_mask_float(indices, shape(x))
    x = x + value * mask
    return x


def assignment(x, values, indices, axis=0):
    if ndim((array(values))) == 0:
        return assignment_single_value(x, values, indices, axis)

    else:
        if ndim(array(indices)) == 0:
            indices = [indices]

        if len(values) != len(indices):
            raise ValueError('Either one value or as many values as indices')

        for (nb_index, index) in enumerate(indices):
            x = assignment_single_value(x, values[nb_index], index, axis)
        return x


def vectorize(x, pyfunc, multiple_args=False, dtype=None, **kwargs):
    if multiple_args:
        return tf.map_fn(lambda x: pyfunc(*x), elems=x, dtype=dtype)
    return tf.map_fn(pyfunc, elems=x, dtype=dtype)


def hsplit(x, n_splits):
    return tf.split(x, num_or_size_splits=n_splits, axis=1)


def amax(x):
    return tf.reduce_max(x)


def real(x):
    return tf.math.real(x)


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


def boolean_mask(x, mask, name='boolean_mask', axis=None):
    return tf.boolean_mask(x, mask, name, axis)


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


def arccosh(x):
    return tf.acosh(x)


def arcsin(x):
    return tf.asin(x)


def arccos(x):
    return tf.acos(x)


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
    n = cast(n, dtype=int32)
    m = cast(m, dtype=int32)
    return tf.eye(num_rows=n, num_columns=m)


def sum(*args, **kwargs):
    return tf.reduce_sum(*args, **kwargs)


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


def diag(a):
    return tf.map_fn(
        lambda x: tf.linalg.tensor_diag(x),
        a)


def arctan2(*args, **kwargs):
    return tf.atan2(*args, **kwargs)


def diagonal(*args, **kwargs):
    return tf.linalg.diag_part(*args)


def mean(x, axis=None):
    return tf.reduce_mean(x, axis)


def cumprod(x, axis=0):
    if axis is None:
        raise NotImplementedError('cumprod is not defined where axis is None')
    else:
        return tf.math.cumprod(x, axis=axis)
