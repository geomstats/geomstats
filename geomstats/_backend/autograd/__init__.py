"""Autograd based computation backend."""

import autograd.numpy as _np
from autograd.numpy import (
    all,
    allclose,
    amax,
    amin,
    any,
    argmax,
    argmin,
    broadcast_arrays,
    broadcast_to,
    clip,
    complex64,
    complex128,
    concatenate,
    conj,
    cross,
    cumprod,
    cumsum,
    diag_indices,
    diagonal,
    einsum,
    empty_like,
    equal,
    expand_dims,
    flip,
    float32,
    float64,
    greater,
    hsplit,
    hstack,
    int32,
    int64,
    isclose,
    isnan,
    kron,
    less,
    less_equal,
    logical_and,
    logical_or,
    maximum,
    mean,
    meshgrid,
    minimum,
    moveaxis,
    ones_like,
    pad,
    prod,
    quantile,
    repeat,
    reshape,
    searchsorted,
    shape,
    sort,
    split,
    stack,
    std,
    sum,
    take,
    tile,
    transpose,
    trapz,
    tril,
    tril_indices,
    triu,
    triu_indices,
    uint8,
    unique,
    vstack,
    where,
    zeros_like,
)
from autograd.scipy.special import erf, gamma, polygamma  # NOQA

from .._shared_numpy import (
    abs,
    angle,
    arange,
    arccos,
    arccosh,
    arcsin,
    arctan2,
    arctanh,
    array_from_sparse,
    assignment,
    assignment_by_sum,
    ceil,
    cos,
    cosh,
    divide,
    dot,
    exp,
    flatten,
    floor,
    from_numpy,
    get_slice,
    log,
    mat_from_diag_triu_tril,
    matmul,
    matvec,
    mod,
    ndim,
    one_hot,
    power,
    ravel_tril_indices,
    real,
    scatter_add,
    set_diag,
    sign,
    sin,
    sinh,
    sqrt,
    squeeze,
    tan,
    tanh,
    to_numpy,
    trace,
    tril_to_vec,
    triu_to_vec,
    vec_to_diag,
    vectorize,
)
from . import autodiff  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA
from ._common import (
    _box_binary_scalar,
    _box_unary_scalar,
    _dyn_update_dtype,
    array,
    as_dtype,
    atol,
    cast,
    convert_to_wider_dtype,
    eye,
    get_default_cdtype,
    get_default_dtype,
    is_array,
    is_bool,
    is_complex,
    is_floating,
    rtol,
    set_default_dtype,
    to_ndarray,
    zeros,
)

ones = _dyn_update_dtype(target=_np.ones)
linspace = _dyn_update_dtype(target=_np.linspace)
empty = _dyn_update_dtype(target=_np.empty)


def imag(x):
    out = _np.imag(x)
    if is_array(x):
        return out

    return array(out)


def copy(x):
    return _np.array(x, copy=True)


def outer(a, b):
    if a.ndim == 2 and b.ndim == 2:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.outer(a, b).reshape(a.shape + b.shape)
    if b.ndim == 2:
        out = out.swapaxes(-3, -2)

    return out
