"""Numpy based computation backend."""

import numpy as np


def hstack(val):
    return np.hstack(val)


def array(val):
    return np.array(val)


def abs(val):
    return np.abs(val)


def zeros(val):
    return np.zeros(val)


def ones(val):
    return np.ones(val)


def all(val):
    return np.all(val)


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


def to_ndarray(element, to_ndim, axis=0):
    element = np.asarray(element)

    if element.ndim == to_ndim - 1:
        element = np.expand_dims(element, axis=axis)
    assert element.ndim == to_ndim
    return element


def sqrt(val):
    return np.sqrt(val)


def norm(val, axis):
    return np.linalg.norm(val, axis=axis)


def rand(*args, **largs):
    return np.random.rand(*args, **largs)


def isclose(a, b):
    return np.isclose(a, b)


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
