"""Tensorflow based computation backend."""

# TODO(johmathe): Reproduce all unit tests with tensorflow backend.

import tensorflow as tf


int32 = tf.int32


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
    return tf.arccosh(x)


def tan(x):
    return tf.tan(x)


def arcsin(x):
    return tf.asin(x)


def arccos(x):
    return tf.acos(x)


def shape(x):
    return tf.shape(x)


def ndim(x):
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


def dot(x, y):
    return tf.reduce_sum(tf.multiply(x, y))


def maximum(x, y):
    return tf.maximum(x, y)


def greater_equal(x, y):
    return tf.greater_equal(x, y)


def equal(x, y):
    return tf.equal(x, y)


def to_ndarray(x, to_ndim, axis=0):
    if ndim(x) == to_ndim - 1:
        x = tf.expand_dims(x, axis=axis)

    return x


def sqrt(x):
    return tf.sqrt(x)


def isclose(x, y, rtol=1e-05, atol=1e-08):
    rhs = tf.constant(atol) + tf.constant(rtol) * tf.abs(y)
    return tf.less_equal(tf.abs(tf.subtract(x, y)), rhs)


def allclose(x, y, rtol=1e-05, atol=1e-08):
    return tf.reduce_all(isclose(x, y, rtol=rtol, atol=atol))


def less_equal(x, y):
    return tf.less_equal(x, y)


def eye(N, M=None):
    return tf.eye(num_rows=N, num_columns=M)


def average(x):
    return tf.reduce_sum(x)


def matmul(x, y):
    return tf.matmul(x, y)


def sum(*args, **kwargs):
    return tf.reduce_sum(*args, **kwargs)


def einsum(equation, *inputs, **kwargs):
    return tf.einsum(equation, *inputs, **kwargs)


def transpose(x):
    return tf.transpose(x)


def squeeze(x):
    return tf.squeeze(x)


def zeros_like(x):
    return tf.zeros_like(x)


def trace(x, **kwargs):
    return tf.trace(x)


def array(x):
    return tf.constant(x)


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
