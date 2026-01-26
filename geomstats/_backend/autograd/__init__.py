"""Autograd based computation backend."""

import functools

import autograd.numpy as _np
from autograd.numpy import (
    asarray,
    broadcast_arrays,
    complex64,
    complex128,
    diag_indices,
    float32,
    float64,
    hsplit,
    int32,
    int64,
    meshgrid,
    pad,
    quantile,
    searchsorted,
    shape,
    sort,
    take,
    tril_indices,
    triu_indices,
    uint8,
    unique,
)

try:
    from autograd.numpy import trapezoid
except ImportError:
    from autograd.numpy import trapz as trapezoid

from autograd.scipy.special import erf, gamma, polygamma  # NOQA

# Import from _array_api - these work across backends
from .._array_api import (
    abs as _api_abs,
    all as _api_all,
    allclose as _api_allclose,
    amax as _api_amax,
    amin as _api_amin,
    any as _api_any,
    arccos as _api_arccos,
    arccosh as _api_arccosh,
    arcsin as _api_arcsin,
    arctan2 as _api_arctan2,
    arctanh as _api_arctanh,
    argmax as _api_argmax,
    argmin as _api_argmin,
    broadcast_to as _api_broadcast_to,
    ceil as _api_ceil,
    clip as _api_clip,
    concatenate as _api_concatenate,
    conj as _api_conj,
    cos as _api_cos,
    cosh as _api_cosh,
    cross as _api_cross,
    cumprod as _api_cumprod,
    cumsum as _api_cumsum,
    diagonal as _api_diagonal,
    dot as _api_dot,
    einsum as _api_einsum,
    empty_like as _api_empty_like,
    equal as _api_equal,
    exp as _api_exp,
    expand_dims as _api_expand_dims,
    flip as _api_flip,
    floor as _api_floor,
    greater as _api_greater,
    hstack as _api_hstack,
    isclose as _api_isclose,
    isnan as _api_isnan,
    kron as _api_kron,
    less as _api_less,
    less_equal as _api_less_equal,
    log as _api_log,
    logical_and as _api_logical_and,
    logical_or as _api_logical_or,
    matmul as _api_matmul,
    maximum as _api_maximum,
    mean as _api_mean,
    minimum as _api_minimum,
    moveaxis as _api_moveaxis,
    ones_like as _api_ones_like,
    outer as _api_outer,
    power as _api_power,
    prod as _api_prod,
    real as _api_real,
    repeat as _api_repeat,
    reshape as _api_reshape,
    sign as _api_sign,
    sin as _api_sin,
    sinh as _api_sinh,
    split as _api_split,
    sqrt as _api_sqrt,
    squeeze as _api_squeeze,
    stack as _api_stack,
    std as _api_std,
    sum as _api_sum,
    tan as _api_tan,
    tanh as _api_tanh,
    tile as _api_tile,
    trace as _api_trace,
    transpose as _api_transpose,
    tril as _api_tril,
    triu as _api_triu,
    vstack as _api_vstack,
    where as _api_where,
    zeros_like as _api_zeros_like,
)

# Import autograd-specific utilities from _shared_numpy
from .._shared_numpy import (
    angle,
    arange,
    array_from_sparse,
    assignment,
    assignment_by_sum,
    divide,
    flatten,
    from_numpy,
    get_slice,
    mat_from_diag_triu_tril,
    matvec,
    mod,
    ndim,
    one_hot,
    ravel_tril_indices,
    scatter_add,
    set_diag,
    to_numpy,
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


# =============================================================================
# Scalar dtype handling wrappers
# =============================================================================


def _wrap_unary_scalar(func):
    """Wrap function to handle scalar inputs with default dtype."""
    @functools.wraps(func)
    def _wrapped(x, *args, **kwargs):
        if isinstance(x, float):
            x = _np.asarray(x, dtype=get_default_dtype())
        elif isinstance(x, complex):
            x = _np.asarray(x, dtype=get_default_cdtype())
        out = func(x, *args, **kwargs)
        if not hasattr(out, "dtype"):
            out = _np.asarray(out, dtype=get_default_dtype())
        return out
    return _wrapped


def _wrap_binary_scalar(func):
    """Wrap binary function to handle scalar inputs with default dtype."""
    @functools.wraps(func)
    def _wrapped(x1, x2, *args, **kwargs):
        if isinstance(x1, float):
            x1 = _np.asarray(x1, dtype=get_default_dtype())
        if isinstance(x2, float):
            x2 = _np.asarray(x2, dtype=get_default_dtype())
        return func(x1, x2, *args, **kwargs)
    return _wrapped


# =============================================================================
# Math functions with scalar handling
# =============================================================================

abs = _wrap_unary_scalar(_api_abs)  # noqa: A001
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
conj = _wrap_unary_scalar(_api_conj)

arctan2 = _wrap_binary_scalar(_api_arctan2)
power = _wrap_binary_scalar(_api_power)


# =============================================================================
# Direct re-exports from _array_api (no scalar handling needed)
# =============================================================================

all = _api_all  # noqa: A001
allclose = _api_allclose
amax = _api_amax
amin = _api_amin
any = _api_any  # noqa: A001
argmax = _api_argmax
argmin = _api_argmin
broadcast_to = _api_broadcast_to
clip = _api_clip
concatenate = _api_concatenate
cross = _api_cross
cumprod = _api_cumprod


def cumsum(x, axis=None, dtype=None):
    """Cumulative sum - autograd doesn't support dtype in gradient."""
    result = _np.cumsum(x, axis=axis)
    if dtype is not None:
        result = result.astype(dtype)
    return result
diagonal = _api_diagonal
dot = _api_dot
einsum = _api_einsum
empty_like = _api_empty_like
equal = _api_equal
expand_dims = _api_expand_dims
flip = _api_flip
greater = _api_greater
hstack = _api_hstack
isclose = _api_isclose
isnan = _api_isnan
kron = _api_kron
less = _api_less
less_equal = _api_less_equal
logical_and = _api_logical_and
logical_or = _api_logical_or
matmul = _api_matmul
maximum = _api_maximum
mean = _api_mean
minimum = _api_minimum
moveaxis = _api_moveaxis
ones_like = _api_ones_like
outer = _api_outer
prod = _api_prod
repeat = _api_repeat
reshape = _api_reshape
split = _api_split
squeeze = _api_squeeze
stack = _api_stack
std = _api_std
sum = _api_sum  # noqa: A001
tile = _api_tile
trace = _api_trace
transpose = _api_transpose
tril = _api_tril
triu = _api_triu
vstack = _api_vstack
where = _api_where
zeros_like = _api_zeros_like


# =============================================================================
# Autograd-specific functions with dtype handling
# =============================================================================

ones = _dyn_update_dtype(target=_np.ones)
linspace = _dyn_update_dtype(target=_np.linspace)
empty = _dyn_update_dtype(target=_np.empty)


def has_autodiff():
    """If allows for automatic differentiation."""
    return True


def imag(x):
    """Imaginary part with array handling."""
    out = _np.imag(x)
    if is_array(x):
        return out
    return array(out)


def copy(x):
    """Copy array."""
    return _np.array(x, copy=True)
