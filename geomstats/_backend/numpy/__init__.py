"""Numpy based computation backend."""

import autograd.numpy as _np
from autograd.numpy import (  # NOQA
    abs,
    all,
    allclose,
    amax,
    amin,
    any,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    arctanh,
    argmax,
    argmin,
    array,
    asarray,
    ceil,
    clip,
    concatenate,
    cos,
    cosh,
    cumsum,
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
    float32,
    float64,
    floor,
    greater,
    hsplit,
    hstack,
    int32,
    int64,
    isclose,
    less,
    less_equal,
    linspace,
    log,
    logical_and,
    logical_or,
    matmul,
    maximum,
    mean,
    meshgrid,
    mod,
    ones,
    ones_like,
    outer,
    repeat,
    reshape,
    shape,
    sign,
    sin,
    sinh,
    split,
    sqrt,
    squeeze,
    stack,
    std,
    sum,
    tan,
    tanh,
    tile,
    trace,
    transpose,
    triu_indices,
    tril_indices,
    diag_indices,
    searchsorted,
    tril,
    vstack,
    where,
    zeros,
    zeros_like
)
from scipy.sparse import coo_matrix

from . import linalg  # NOQA
from . import random  # NOQA


# XXX(nkoep): Can we get rid of this now?
def while_loop(cond, body, loop_vars, maximum_iterations):
    iteration = 0
    while cond(*loop_vars):
        loop_vars = body(*loop_vars)
        iteration += 1
        if iteration >= maximum_iterations:
            break
    return loop_vars


def flatten(x):
    return x.flatten()


def get_mask_i_float(i, n):
    range_n = arange(n)
    i_float = cast(array([i]), int32)[0]
    mask_i = equal(range_n, i_float)
    mask_i_float = cast(mask_i, float32)
    return mask_i_float


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
    x_new = copy(x)
    if not isinstance(indices, list):
        indices = [indices]
    if not isinstance(values, list):
        values = [values] * len(indices)
    for nb_index, index in enumerate(indices):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) < len(shape(x)):
            for n_axis in range(shape(x)[axis]):
                extended_index = index[:axis] + (n_axis,) + index[axis:]
                x_new[extended_index] = values[nb_index]
        else:
            x_new[index] = values[nb_index]
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
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)
    if not isinstance(indices, list):
        indices = [indices]
    if not isinstance(values, list):
        values = [values] * len(indices)
    for nb_index, index in enumerate(indices):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) < len(shape(x)):
            for n_axis in range(shape(x)[axis]):
                extended_index = index[:axis] + (n_axis,) + index[axis:]
                x_new[extended_index] += values[nb_index]
        else:
            x_new[index] += values[nb_index]
    return x_new


def gather(x, indices, axis=0):
    return x[indices]


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    if multiple_args:
        return _np.vectorize(pyfunc, signature=signature)(*x)
    return _np.vectorize(pyfunc, signature=signature)(x)


# XXX(nkoep): Can we get rid of this now?
def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def cast(x, dtype):
    return x.astype(dtype)


def to_ndarray(x, to_ndim, axis=0):
    x = _np.array(x)
    if x.ndim == to_ndim - 1:
        x = _np.expand_dims(x, axis=axis)

    if x.ndim != 0:
        assert x.ndim >= to_ndim
    return x


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
        result[i] = aux[i * n:(i + 1) * n, i * n:(i + 1) * n]
    return result


def eval(x):
    return x


def ndim(x):
    return x.ndim


def copy(x):
    return x.copy()


def array_from_sparse(indices, data, target_shape):
    return array(
        coo_matrix((data, list(zip(*indices))), target_shape).todense())


def from_vector_to_diagonal_matrix(x):
    n = shape(x)[-1]
    identity_n = eye(n)
    diagonals = einsum('ki,ij->kij', x, identity_n)
    return diagonals
