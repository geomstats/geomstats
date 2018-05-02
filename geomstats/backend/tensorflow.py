"""Tensorflow based computation backend."""

# TODO(johmathe): Reproduce all unit tests with tensorflow backend.

import tensorflow as tf

pi = tf.pi


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


def cosh(*args, **kwargs):
    return tf.cosh(*args, **kwargs)


def sinh(*args, **kwargs):
    return tf.sinh(*args, **kwargs)


def tanh(*args, **kwargs):
    return tf.tanh(*args, **kwargs)


def arccosh(*args, **kwargs):
    return tf.arccosh(*args, **kwargs)


def tan(x):
    return tf.tan(x)


def arcsin(x):
    return tf.asin(x)


def arccos(x):
    return tf.acos(x)


def shape(x):
    return tf.shape(x)


def dot(x, y):
    return tf.reduce_sum(tf.multiply(x, y))


def maximum(x, y):
    return tf.maximum(x, y)


def greater_equal(x, y):
    return tf.greater_equal(x, y)


def to_ndarray(element, to_ndim, axis=0):

    if element.ndim == to_ndim - 1:
        element = tf.expand_dims(element, axis=axis)
    assert element.ndim == to_ndim
    return element


def sqrt(x):
    return tf.sqrt(x)


def norm(x, axis=None, keepdims=None):
    return tf.linalg.norm(x, axis=axis, keepdims=keepdims)


def isclose(x, y, rtol=1e-05, atol=1e-08):
    rhs = tf.constant(atol) + tf.constant(rtol) * tf.abs(y)
    return tf.less_equal(tf.abs(tf.sub(x, y)), rhs)


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
