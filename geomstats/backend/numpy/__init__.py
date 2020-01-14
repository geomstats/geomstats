"""Numpy based computation backend."""

import numpy as _np

from . import linalg  # NOQA
from . import random  # NOQA
from . import testing  # NOQA


int32 = _np.int32
int8 = _np.int8
float32 = _np.float32
float64 = _np.float64


def indexing(x):
    return x


def float_to_double(x):
    return x


def byte_to_float(x):
    return x


def while_loop(cond, body, loop_vars, maximum_iterations):
    iteration = 0
    while cond(*loop_vars):
        loop_vars = body(*loop_vars)
        iteration += 1
        if iteration >= maximum_iterations:
            break
    return loop_vars


def logical_or(x, y):
    bool_result = x or y
    return bool_result


def get_mask_i_float(i, n):
    range_n = arange(n)
    i_float = cast(array([i]), int32)[0]
    mask_i = equal(range_n, i_float)
    mask_i_float = cast(mask_i, float32)
    return mask_i_float


def gather(x, indices, axis=0):
    return x[indices]


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    if multiple_args:
        return _np.vectorize(pyfunc, signature=signature)(*x)
    return _np.vectorize(pyfunc, signature=signature)(x)


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def real(x):
    return _np.real(x)


def reshape(*args, **kwargs):
    return _np.reshape(*args, **kwargs)


def cast_to_complex(x):
    return _np.vectorize(complex)(x)


def boolean_mask(x, mask):
    return x[mask]


def flip(*args, **kwargs):
    return _np.flip(*args, **kwargs)


def amax(*args, **kwargs):
    return _np.amax(*args, **kwargs)


def arctan2(*args, **kwargs):
    return _np.arctan2(*args, **kwargs)


def cast(x, dtype):
    return x.astype(dtype)


def divide(*args, **kwargs):
    return _np.divide(*args, **kwargs)


def repeat(*args, **kwargs):
    return _np.repeat(*args, **kwargs)


def asarray(*args, **kwargs):
    return _np.asarray(*args, **kwargs)


def concatenate(*args, **kwargs):
    return _np.concatenate(*args, **kwargs)


def identity(val):
    return _np.identity(val)


def hstack(val):
    return _np.hstack(val)


def stack(*args, **kwargs):
    return _np.stack(*args, **kwargs)


def vstack(val):
    return _np.vstack(val)


def array(val):
    return _np.array(val)


def abs(val):
    return _np.abs(val)


def zeros(val):
    return _np.zeros(val)


def ones(val):
    return _np.ones(val)


def ones_like(*args, **kwargs):
    return _np.ones_like(*args, **kwargs)


def empty_like(*args, **kwargs):
    return _np.empty_like(*args, **kwargs)


def all(*args, **kwargs):
    return _np.all(*args, **kwargs)


def allclose(a, b, **kwargs):
    return _np.allclose(a, b, **kwargs)


def sin(val):
    return _np.sin(val)


def cos(val):
    return _np.cos(val)


def cosh(*args, **kwargs):
    return _np.cosh(*args, **kwargs)


def sinh(*args, **kwargs):
    return _np.sinh(*args, **kwargs)


def tanh(*args, **kwargs):
    return _np.tanh(*args, **kwargs)


def arccosh(*args, **kwargs):
    return _np.arccosh(*args, **kwargs)


def tan(val):
    return _np.tan(val)


def arcsin(val):
    return _np.arcsin(val)


def arccos(val):
    return _np.arccos(val)


def shape(val):
    return val.shape


def dot(a, b):
    return _np.dot(a, b)


def maximum(a, b):
    return _np.maximum(a, b)


def greater(a, b):
    return _np.greater(a, b)


def greater_equal(a, b):
    return _np.greater_equal(a, b)


def to_ndarray(x, to_ndim, axis=0):
    x = _np.asarray(x)
    if x.ndim == to_ndim - 1:
        x = _np.expand_dims(x, axis=axis)
    assert x.ndim >= to_ndim
    return x


def sqrt(val):
    return _np.sqrt(val)


def norm(val, axis):
    return _np.linalg.norm(val, axis=axis)


def rand(*args, **largs):
    return _np.random.rand(*args, **largs)


def randint(*args, **kwargs):
    return _np.random.randint(*args, **kwargs)


def isclose(*args, **kwargs):
    return _np.isclose(*args, **kwargs)


def less_equal(a, b):
    return _np.less_equal(a, b)


def less(a, b):
    return _np.less(a, b)


def eye(*args, **kwargs):
    return _np.eye(*args, **kwargs)


def average(*args, **kwargs):
    return _np.average(*args, **kwargs)


def matmul(*args, **kwargs):
    return _np.matmul(*args, **kwargs)


def sum(*args, **kwargs):
    return _np.sum(*args, **kwargs)


def einsum(*args, **kwargs):
    return _np.einsum(*args, **kwargs)


def transpose(*args, **kwargs):
    return _np.transpose(*args, **kwargs)


def squeeze(*args, **kwargs):
    return _np.squeeze(*args, **kwargs)


def zeros_like(*args, **kwargs):
    return _np.zeros_like(*args, **kwargs)


def trace(*args, **kwargs):
    return _np.trace(*args, **kwargs)


def mod(*args, **kwargs):
    return _np.mod(*args, **kwargs)


def linspace(*args, **kwargs):
    return _np.linspace(*args, **kwargs)


def equal(*args, **kwargs):
    return _np.equal(*args, **kwargs)


def floor(*args, **kwargs):
    return _np.floor(*args, **kwargs)


def cross(*args, **kwargs):
    return _np.cross(*args, **kwargs)


def triu_indices(*args, **kwargs):
    return _np.triu_indices(*args, **kwargs)


def where(*args, **kwargs):
    return _np.where(*args, **kwargs)


def tile(*args, **kwargs):
    return _np.tile(*args, **kwargs)


def clip(*args, **kwargs):
    return _np.clip(*args, **kwargs)


def diag(x):
    x = to_ndarray(x, to_ndim=2)
    _, n = shape(x)
    aux = _np.vectorize(
        _np.diagflat,
        signature='(m,n)->(k,k)')(x)
    k, k = shape(aux)
    m = int(k / n)
    result = zeros((m, n, n))
    for i in range(m):
        result[i] = aux[i*n:(i+1)*n, i*n:(i+1)*n]
    return result


def any(*args, **kwargs):
    return _np.any(*args, **kwargs)


def expand_dims(*args, **kwargs):
    return _np.expand_dims(*args, **kwargs)


def outer(*args, **kwargs):
    return _np.outer(*args, **kwargs)


def hsplit(*args, **kwargs):
    return _np.hsplit(*args, **kwargs)


def argmax(*args, **kwargs):
    return _np.argmax(*args, **kwargs)


def argmin(*args, **kwargs):
    return _np.argmin(*args, **kwargs)


def diagonal(*args, **kwargs):
    return _np.diagonal(*args, **kwargs)


def exp(*args, **kwargs):
    return _np.exp(*args, **kwargs)


def log(*args, **kwargs):
    return _np.log(*args, **kwargs)


def cov(*args, **kwargs):
    return _np.cov(*args, **kwargs)


def eval(x):
    return x


def ndim(x):
    return x.ndim


def nonzero(x):
    return _np.nonzero(x)


def copy(x):
    return _np.copy(x)


def ix_(*args):
    return _np.ix_(*args)


def arange(*args, **kwargs):
    return _np.arange(*args, **kwargs)


def prod(x, axis=None):
    return _np.prod(x, axis=axis)


def sign(*args, **kwargs):
    return _np.sign(*args, **kwargs)


def mean(x, axis=None):
    return _np.mean(x, axis)


def normal(*args, **kwargs):
    return _np.random.normal(*args, **kwargs)
