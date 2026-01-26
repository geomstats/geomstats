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
    asarray,
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

try:
    from autograd.numpy import trapezoid
except ImportError:
    from autograd.numpy import trapz as trapezoid

from autograd.scipy.special import erf, gamma, polygamma  # NOQA

from .._array_api import (
    abs as _api_abs,
    arccos as _api_arccos,
    arccosh as _api_arccosh,
    arcsin as _api_arcsin,
    arctan2 as _api_arctan2,
    arctanh as _api_arctanh,
    ceil as _api_ceil,
    cos as _api_cos,
    cosh as _api_cosh,
    exp as _api_exp,
    floor as _api_floor,
    log as _api_log,
    power as _api_power,
    real as _api_real,
    sign as _api_sign,
    sin as _api_sin,
    sinh as _api_sinh,
    sqrt as _api_sqrt,
    tan as _api_tan,
    tanh as _api_tanh,
)
from .._shared_numpy import (
    angle,
    arange,
    array_from_sparse,
    assignment,
    assignment_by_sum,
    divide,
    dot,
    flatten,
    from_numpy,
    get_slice,
    mat_from_diag_triu_tril,
    matmul,
    matvec,
    mod,
    ndim,
    one_hot,
    ravel_tril_indices,
    scatter_add,
    set_diag,
    squeeze,
    to_numpy,
    trace,
    tril_to_vec,
    triu_to_vec,
    vec_to_diag,
    vectorize,
)
from . import (
    autodiff,  # NOQA
    linalg,  # NOQA
    random,  # NOQA
)
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

def _wrap_unary_scalar(func):
    """Wrap _array_api function to handle scalar inputs with default dtype."""
    import functools

    @functools.wraps(func)
    def _wrapped(x, *args, **kwargs):
        if isinstance(x, float):
            x = _np.asarray(x, dtype=get_default_dtype())
        elif isinstance(x, complex):
            x = _np.asarray(x, dtype=get_default_cdtype())
        out = func(x, *args, **kwargs)
        # Ensure output is an array with proper dtype
        if not hasattr(out, "dtype"):
            out = _np.asarray(out, dtype=get_default_dtype())
        return out

    return _wrapped


def _wrap_binary_scalar(func):
    """Wrap _array_api binary function to handle scalar inputs with default dtype."""
    import functools

    @functools.wraps(func)
    def _wrapped(x1, x2, *args, **kwargs):
        if isinstance(x1, float):
            x1 = _np.asarray(x1, dtype=get_default_dtype())
        if isinstance(x2, float):
            x2 = _np.asarray(x2, dtype=get_default_dtype())
        return func(x1, x2, *args, **kwargs)

    return _wrapped


# Wrap _array_api functions with scalar dtype handling
abs = _wrap_unary_scalar(_api_abs)
arccos = _wrap_unary_scalar(_api_arccos)
arccosh = _wrap_unary_scalar(_api_arccosh)
arcsin = _wrap_unary_scalar(_api_arcsin)
arctanh = _wrap_unary_scalar(_api_arctanh)
ceil = _wrap_unary_scalar(_api_ceil)
cos = _wrap_unary_scalar(_api_cos)
cosh = _wrap_unary_scalar(_api_cosh)
exp = _wrap_unary_scalar(_api_exp)
floor = _wrap_unary_scalar(_api_floor)
log = _wrap_unary_scalar(_api_log)
real = _wrap_unary_scalar(_api_real)
sign = _wrap_unary_scalar(_api_sign)
sin = _wrap_unary_scalar(_api_sin)
sinh = _wrap_unary_scalar(_api_sinh)
sqrt = _wrap_unary_scalar(_api_sqrt)
tan = _wrap_unary_scalar(_api_tan)
tanh = _wrap_unary_scalar(_api_tanh)

arctan2 = _wrap_binary_scalar(_api_arctan2)
power = _wrap_binary_scalar(_api_power)

ones = _dyn_update_dtype(target=_np.ones)
linspace = _dyn_update_dtype(target=_np.linspace)
empty = _dyn_update_dtype(target=_np.empty)


def has_autodiff():
    """If allows for automatic differentiation.

    Returns
    -------
    has_autodiff : bool
    """
    return True


def imag(x):
    out = _np.imag(x)
    if is_array(x):
        return out

    return array(out)


def copy(x):
    return _np.array(x, copy=True)


def outer(a, b):
    if a.ndim > 1 and b.ndim > 1:
        return _np.einsum("...i,...j->...ij", a, b)

    out = _np.outer(a, b).reshape(a.shape + b.shape)
    if b.ndim > 1:
        out = out.swapaxes(0, -2)

    return out
