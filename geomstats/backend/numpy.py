"""Numpy based computation backend."""

import numpy as np

int32 = np.int32
int8 = np.int8
double = np.float32
float64 = np.float64


def flip(*args, **kwargs):
    return np.flip(*args, **kwargs)


def amax(*args, **kwargs):
    return np.amax(*args, **kwargs)


def arctan2(*args, **kwargs):
    return np.arctan2(*args, **kwargs)


def cast(x, dtype):
    return x.astype(dtype)


def divide(*args, **kwargs):
    return np.divide(*args, **kwargs)


def repeat(*args, **kwargs):
    return np.repeat(*args, **kwargs)


def asarray(*args, **kwargs):
    return np.asarray(*args, **kwargs)


def concatenate(*args, **kwargs):
    return np.concatenate(*args, **kwargs)


def identity(val):
    return np.identity(val)


def hstack(val):
    return np.hstack(val)


def stack(*args, **kwargs):
    return np.stack(*args, **kwargs)


def vstack(val):
    return np.vstack(val)


def array(val):
    return np.array(val)


def abs(val):
    return np.abs(val)


def zeros(val):
    return np.zeros(val)


def ones(val):
    return np.ones(val)


def ones_like(*args, **kwargs):
    return np.ones_like(*args, **kwargs)


def empty_like(*args, **kwargs):
    return np.empty_like(*args, **kwargs)


def all(*args, **kwargs):
    return np.all(*args, **kwargs)


def allclose(a, b, **kwargs):
    return np.allclose(a, b, **kwargs)


def sin(val):
    return np.sin(val)


def cos(val):
    return np.cos(val)


def cosh(*args, **kwargs):
    return np.cosh(*args, **kwargs)


def sinh(*args, **kwargs):
    return np.sinh(*args, **kwargs)


def tanh(*args, **kwargs):
    return np.tanh(*args, **kwargs)


def arccosh(*args, **kwargs):
    return np.arccosh(*args, **kwargs)


def tan(val):
    return np.tan(val)


def arcsin(val):
    return np.arcsin(val)


def arccos(val):
    return np.arccos(val)


def shape(val):
    return val.shape


def dot(a, b):
    return np.dot(a, b)


def maximum(a, b):
    return np.maximum(a, b)


def greater_equal(a, b):
    return np.greater_equal(a, b)


def to_ndarray(x, to_ndim, axis=0):
    x = np.asarray(x)
    if x.ndim == to_ndim - 1:
        x = np.expand_dims(x, axis=axis)
    assert x.ndim >= to_ndim
    return x


def sqrt(val):
    return np.sqrt(val)


def norm(val, axis):
    return np.linalg.norm(val, axis=axis)


def rand(*args, **largs):
    return np.random.rand(*args, **largs)


def isclose(*args, **kwargs):
    return np.isclose(*args, **kwargs)


def less_equal(a, b):
    return np.less_equal(a, b)


def eye(*args, **kwargs):
    return np.eye(*args, **kwargs)


def average(*args, **kwargs):
    return np.average(*args, **kwargs)


def matmul(*args, **kwargs):
    return np.matmul(*args, **kwargs)


def sum(*args, **kwargs):
    return np.sum(*args, **kwargs)


def einsum(*args, **kwargs):
    return np.einsum(*args, **kwargs)


def transpose(*args, **kwargs):
    return np.transpose(*args, **kwargs)


def squeeze(*args, **kwargs):
    return np.squeeze(*args, **kwargs)


def zeros_like(*args, **kwargs):
    return np.zeros_like(*args, **kwargs)


def trace(*args, **kwargs):
    return np.trace(*args, **kwargs)


def mod(*args, **kwargs):
    return np.mod(*args, **kwargs)


def linspace(*args, **kwargs):
    return np.linspace(*args, **kwargs)


def equal(*args, **kwargs):
    return np.equal(*args, **kwargs)


def floor(*args, **kwargs):
    return np.floor(*args, **kwargs)


def cross(*args, **kwargs):
    return np.cross(*args, **kwargs)


def triu_indices(*args, **kwargs):
    return np.triu_indices(*args, **kwargs)


def where(*args, **kwargs):
    return np.where(*args, **kwargs)


def tile(*args, **kwargs):
    return np.tile(*args, **kwargs)


def clip(*args, **kwargs):
    return np.clip(*args, **kwargs)


def diag(*args, **kwargs):
    return np.diag(*args, **kwargs)


def any(*args, **kwargs):
    return np.any(*args, **kwargs)


def expand_dims(*args, **kwargs):
    return np.expand_dims(*args, **kwargs)


def outer(*args, **kwargs):
    return np.outer(*args, **kwargs)


def hsplit(*args, **kwargs):
    return np.hsplit(*args, **kwargs)


def argmax(*args, **kwargs):
    return np.argmax(*args, **kwargs)


def diagonal(*args, **kwargs):
    return np.diagonal(*args, **kwargs)


def exp(*args, **kwargs):
    return np.exp(*args, **kwargs)


def log(*args, **kwargs):
    return np.log(*args, **kwargs)


def cov(*args, **kwargs):
    return np.cov(*args, **kwargs)


def eval(x):
    return x


def ndim(x):
    return x.ndim


def nonzero(x):
    return np.nonzero(x)


def copy(x):
    return np.copy(x)
