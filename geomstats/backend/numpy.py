"""Numpy based computation backend."""

import numpy as np
from numpy import (  # NOQA
    abs,
    all,
    allclose,
    amax,
    amin,
    any,
    append,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    argmax,
    argmin,
    array,
    asarray,
    average,
    clip,
    concatenate,
    cos,
    cosh,
    cov,
    cross,
    diagonal,
    divide,
    dot,
    einsum,
    empty,
    empty_like,
    equal,
    exp,
    expand_dims,
    eye,
    flip,
    floor,
    greater_equal,
    greater,
    hsplit,
    hstack,
    identity,
    isclose,
    ix_,
    less,
    less_equal,
    linspace,
    log,
    matmul,
    maximum,
    mean,
    mod,
    nonzero,
    ones,
    ones_like,
    outer,
    prod,
    real,
    repeat,
    reshape,
    shape,
    sign,
    sin,
    sinh,
    sqrt,
    squeeze,
    stack,
    sum,
    tan,
    tanh,
    tile,
    trace,
    transpose,
    triu_indices,
    vstack,
    where,
    zeros,
    zeros_like
)


int32 = np.int32
int8 = np.int8
float32 = np.float32
float64 = np.float64


def copy(x):
    return x.copy()


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
        return np.vectorize(pyfunc, signature=signature)(*x)
    return np.vectorize(pyfunc, signature=signature)(x)


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def cast_to_complex(x):
    return np.vectorize(complex)(x)


def boolean_mask(x, mask):
    return x[mask]


def cast(x, dtype):
    return x.astype(dtype)


def to_ndarray(x, to_ndim, axis=0):
    x = np.asarray(x)
    if x.ndim == to_ndim - 1:
        x = np.expand_dims(x, axis=axis)
    assert x.ndim >= to_ndim
    return x


def diag(x):
    x = to_ndarray(x, to_ndim=2)
    _, n = shape(x)
    aux = np.vectorize(
        np.diagflat,
        signature='(m,n)->(k,k)')(x)
    k, k = shape(aux)
    m = int(k / n)
    result = zeros((m, n, n))
    for i in range(m):
        result[i] = aux[i*n:(i+1)*n, i*n:(i+1)*n]
    return result


def eval(x):
    return x


def ndim(x):
    return x.ndim


def cumprod(x, axis=0):
    if axis is None:
        raise NotImplementedError('cumprod is not defined where axis is None')
    return np.cumprod(x, axis=axis)
