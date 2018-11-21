"""Tensorflow based computation backend."""

# TODO(johmathe): Reproduce all unit tests with tensorflow backend.

import tensorflow as tf


int8 = tf.int8
int32 = tf.int32
int64 = tf.int64
float16 = tf.float16
float32 = tf.float32
float64 = tf.float64


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
    return tf.log(x)


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


def ndim(x):
    x = array(x)
    dims = x.get_shape()._dims
    if dims is not None:
        return len(dims)
    return None


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


def less(x, y):
    return tf.less(x, y)


def less_equal(x, y):
    return tf.less_equal(x, y)


def eye(N, M=None):
    return tf.eye(num_rows=N, num_columns=M)


def matmul(x, y):
    return tf.matmul(x, y)


def argmax(*args, **kwargs):
    return tf.argmax(*args, **kwargs)


def sum(*args, **kwargs):
    return tf.reduce_sum(*args, **kwargs)


def einsum(equation, *inputs, **kwargs):
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
    return tf.trace(x)


def array(x):
    return tf.convert_to_tensor(x)


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
    return tf.diag(a)


def cross(a, b):
    return tf.cross(a, b)


def stack(*args, **kwargs):
    return tf.stack(*args, **kwargs)


def arctan2(*args, **kwargs):
    return tf.atan2(*args, **kwargs)


def diagonal(x):
    return tf.linalg.diag_part(x)


def mean(x, axis=None):
    return tf.reduce_mean(x, axis)
