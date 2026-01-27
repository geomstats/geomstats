"""Pytorch based computation backend."""

import functools
from collections.abc import Iterable as _Iterable

import numpy as _np
import torch as _torch
from torch import (
    asarray,
    complex64,
    complex128,
    empty,
    erf,
    float32,
    float64,
    int32,
    int64,
    meshgrid,
    ones,
    polygamma,
    quantile,
    searchsorted,
    trapezoid,
    uint8,
)
from torch import broadcast_tensors as broadcast_arrays
from torch.special import gammaln as _gammaln

from .._backend_config import pytorch_atol as atol
from .._backend_config import pytorch_rtol as rtol

# Import from _array_api - these work across backends
from .._array_api import (
    abs as _api_abs,
    amax as _api_amax,
    amin as _api_amin,
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
    cumprod as _api_cumprod,
    cumsum as _api_cumsum,
    diagonal as _api_diagonal,
    empty_like as _api_empty_like,
    exp as _api_exp,
    expand_dims as _api_expand_dims,
    flip as _api_flip,
    floor as _api_floor,
    greater as _api_greater,
    hstack as _api_hstack,
    isnan as _api_isnan,
    kron as _api_kron,
    less as _api_less,
    log as _api_log,
    logical_or as _api_logical_or,
    maximum as _api_maximum,
    mean as _api_mean,
    minimum as _api_minimum,
    moveaxis as _api_moveaxis,
    ones_like as _api_ones_like,
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
    transpose as _api_transpose,
    tril as _api_tril,
    triu as _api_triu,
    vstack as _api_vstack,
    zeros_like as _api_zeros_like,
)

from . import (
    autodiff,  # NOQA
    linalg,  # NOQA
    random,  # NOQA
)
from ._common import array, cast, from_numpy
from ._dtype import (
    _add_default_dtype_by_casting,
    _preserve_input_dtype,
    as_dtype,
    get_default_cdtype,
    get_default_dtype,
    is_bool,
    is_complex,
    is_floating,
    set_default_dtype,
)


# =============================================================================
# Dtype handling
# =============================================================================

_DTYPES = {
    int32: 0,
    int64: 1,
    float32: 2,
    float64: 3,
    complex64: 4,
    complex128: 5,
}


def convert_to_wider_dtype(tensor_list):
    """Convert tensors to common wider dtype."""
    dtype_list = [_DTYPES.get(x.dtype, -1) for x in tensor_list]
    if len(set(dtype_list)) == 1:
        return tensor_list
    wider_dtype_index = max(dtype_list)
    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]
    return [cast(x, dtype=wider_dtype) for x in tensor_list]


# =============================================================================
# Tensor wrapping helpers
# =============================================================================


def _wrap_unary_scalar(func):
    """Wrap function to convert scalars to tensors."""
    @functools.wraps(func)
    def _wrapped(x, *args, **kwargs):
        if not _torch.is_tensor(x):
            x = _torch.tensor(x)
        return func(x, *args, **kwargs)
    return _wrapped


def _wrap_binary_scalar(func):
    """Wrap binary function to convert scalars to tensors."""
    @functools.wraps(func)
    def _wrapped(x1, x2, *args, **kwargs):
        if not _torch.is_tensor(x1):
            x1 = _torch.tensor(x1)
        if not _torch.is_tensor(x2):
            x2 = _torch.tensor(x2)
        return func(x1, x2, *args, **kwargs)
    return _wrapped


# =============================================================================
# Math functions with tensor wrapping
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
# Direct re-exports from _array_api
# =============================================================================

amax = _api_amax
amin = _api_amin
argmax = _api_argmax
argmin = _api_argmin
broadcast_to = _api_broadcast_to
clip = _api_clip
cumprod = _api_cumprod
cumsum = _api_cumsum
diagonal = _api_diagonal
empty_like = _api_empty_like
expand_dims = _api_expand_dims
flip = _api_flip
greater = _api_greater
hstack = _api_hstack
isnan = _api_isnan
kron = _api_kron
less = _api_less
logical_or = _api_logical_or
maximum = _api_maximum
mean = _api_mean
minimum = _api_minimum
moveaxis = _api_moveaxis
ones_like = _api_ones_like
prod = _api_prod
repeat = _api_repeat
reshape = _api_reshape
squeeze = _api_squeeze
stack = _api_stack
std = _wrap_unary_scalar(_preserve_input_dtype(_add_default_dtype_by_casting(target=_torch.std)))
tile = _api_tile
transpose = _api_transpose
tril = _api_tril
triu = _api_triu
vstack = _api_vstack
zeros = _torch.zeros
zeros_like = _api_zeros_like


# =============================================================================
# PyTorch-specific implementations
# =============================================================================


def has_autodiff():
    """If allows for automatic differentiation."""
    return True


def angle(x, deg=False):
    """Return the angle of the complex argument."""
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    return _torch.angle(x)


def mod(x1, x2):
    """Element-wise remainder of division."""
    if not _torch.is_tensor(x1):
        x1 = _torch.tensor(x1)
    return _torch.remainder(x1, x2)


def gamma(a):
    """Gamma function."""
    return _torch.exp(_gammaln(a))


def imag(a):
    """Imaginary part of complex argument."""
    if not _torch.is_tensor(a):
        a = _torch.tensor(a)
    if is_complex(a):
        return _torch.imag(a)
    return _torch.zeros(a.shape, dtype=a.dtype)


# =============================================================================
# Functions with dtype conversion
# =============================================================================


def matmul(x, y, out=None):
    """Matrix multiplication with dtype handling."""
    for arr in [x, y]:
        if arr.ndim == 1:
            raise ValueError("ndims must be >=2")
    x, y = convert_to_wider_dtype([x, y])
    return _torch.matmul(x, y, out=out)


def cross(a, b):
    """Cross product with dtype handling."""
    if a.shape != b.shape:
        a, b = broadcast_arrays(a, b)
    return _torch.cross(*convert_to_wider_dtype([a, b]), dim=-1)


def einsum(equation, *inputs):
    """Einstein summation with dtype handling."""
    tensors = [arg if is_array(arg) else array(arg) for arg in inputs]
    tensors = convert_to_wider_dtype(tensors)
    return _torch.einsum(equation, *tensors)


def concatenate(seq, axis=0, out=None):
    """Concatenate with dtype handling."""
    seq = convert_to_wider_dtype(list(seq))
    return _torch.cat(seq, dim=axis, out=out)


def split(x, indices_or_sections, axis=0):
    """Split array."""
    if isinstance(indices_or_sections, int):
        split_size = x.shape[axis] // indices_or_sections
        return _torch.split(x, split_size, dim=axis)
    indices = _np.asarray(indices_or_sections)
    sizes = []
    prev = 0
    for idx in indices:
        sizes.append(idx - prev)
        prev = idx
    sizes.append(x.shape[axis] - prev)
    return _torch.split(x, sizes, dim=axis)


# =============================================================================
# Functions with custom implementations
# =============================================================================


def sum(x, axis=None, keepdims=None, dtype=None):  # noqa: A001
    """Sum with axis/keepdims handling."""
    if axis is None:
        if keepdims is None:
            return _torch.sum(x, dtype=dtype)
        return _torch.sum(x, keepdim=keepdims, dtype=dtype)
    if keepdims is None:
        return _torch.sum(x, dim=axis, dtype=dtype)
    return _torch.sum(x, dim=axis, keepdim=keepdims, dtype=dtype)


def all(x, axis=None):  # noqa: A001
    """Test if all elements are true."""
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if axis is None:
        return x.bool().all()
    if isinstance(axis, int):
        return _torch.all(x.bool(), axis)
    axis = list(axis)
    for i, a in enumerate(axis):
        if a < 0:
            axis[i] = ndim(x) + a
    result = x
    for a in sorted(axis, reverse=True):
        result = _torch.all(result.bool(), a)
    return result


def any(x, axis=None):  # noqa: A001
    """Test if any element is true."""
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if axis is None:
        return _torch.any(x)
    if isinstance(axis, int):
        return _torch.any(x.bool(), axis)
    axis = list(axis)
    for i, a in enumerate(axis):
        if a < 0:
            axis[i] = ndim(x) + a
    result = x
    for a in sorted(axis, reverse=True):
        result = _torch.any(result.bool(), a)
    return result


def logical_and(x, y):
    """Element-wise logical AND."""
    if _torch.is_tensor(x) or _torch.is_tensor(y):
        return x * y
    return x and y


def equal(a, b, **kwargs):
    """Element-wise equality."""
    if not is_array(a):
        a = array(a)
    if not is_array(b):
        b = array(b)
    return _torch.eq(a, b, **kwargs)


def less_equal(x, y, **kwargs):
    """Element-wise less-than-or-equal."""
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    return _torch.le(x, y, **kwargs)


def where(condition, x=None, y=None):
    """Return elements chosen from x or y."""
    if not _torch.is_tensor(condition):
        condition = array(condition)
    if x is None and y is None:
        return _torch.where(condition)
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    y = cast(y, x.dtype)
    return _torch.where(condition, x, y)


def allclose(a, b, atol=atol, rtol=rtol):
    """Test if all elements are approximately equal."""
    if not _torch.is_tensor(a):
        a = _torch.tensor(a)
    if not _torch.is_tensor(b):
        b = _torch.tensor(b)
    a, b = _torch.broadcast_tensors(a, b)
    return _torch.allclose(a, b, atol=atol, rtol=rtol)


def isclose(x, y, rtol=rtol, atol=atol):
    """Element-wise test for approximate equality."""
    if not _torch.is_tensor(x):
        x = _torch.tensor(x)
    if not _torch.is_tensor(y):
        y = _torch.tensor(y)
    return _torch.isclose(x, y, atol=atol, rtol=rtol)


# =============================================================================
# Array creation and indexing
# =============================================================================


def arange(start, stop=None, step=1, dtype=None):
    """Return evenly spaced values."""
    if stop is None:
        return _torch.arange(start, dtype=dtype)
    return _torch.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    """Return evenly spaced numbers."""
    start_is_tensor = _torch.is_tensor(start)
    stop_is_tensor = _torch.is_tensor(stop)
    if not (start_is_tensor or stop_is_tensor) and endpoint:
        return _torch.linspace(start=start, end=stop, steps=num, dtype=dtype)
    if not start_is_tensor:
        start = _torch.tensor(start)
    if not stop_is_tensor:
        stop = _torch.tensor(stop)
    start, stop = _torch.broadcast_tensors(start, stop)
    result_shape = (num, *start.shape)
    start_flat = _torch.flatten(start)
    stop_flat = _torch.flatten(stop)
    if endpoint:
        result = _torch.vstack([
            _torch.linspace(start=start_flat[i], end=stop_flat[i], steps=num, dtype=dtype)
            for i in range(start_flat.shape[0])
        ]).T
    else:
        result = _torch.vstack([
            _torch.arange(start_flat[i], stop_flat[i], (stop_flat[i] - start_flat[i]) / num, dtype=dtype)
            for i in range(start_flat.shape[0])
        ]).T
    return _torch.reshape(result, result_shape)


def eye(n, m=None, dtype=None):
    """Return identity matrix."""
    if m is None:
        m = n
    return _torch.eye(n, m, dtype=dtype)


def diag_indices(n, ndim=2):
    """Return indices for diagonal elements."""
    return tuple(map(_torch.from_numpy, _np.diag_indices(n, ndim)))


def tril_indices(n, k=0, m=None):
    """Return indices for lower triangle."""
    if m is None:
        m = n
    return _torch.tril_indices(row=n, col=m, offset=k)


def triu_indices(n, k=0, m=None):
    """Return indices for upper triangle."""
    if m is None:
        m = n
    return _torch.triu_indices(row=n, col=m, offset=k)


# =============================================================================
# Utility functions
# =============================================================================


def is_array(x):
    """Check if x is a tensor."""
    return _torch.is_tensor(x)


def ndim(x):
    """Return number of dimensions."""
    return x.dim()


def shape(val):
    """Return shape of array."""
    if not is_array(val):
        val = array(val)
    return val.shape


def flatten(x):
    """Flatten array."""
    return _torch.flatten(x)


def copy(x):
    """Copy tensor."""
    return x.clone()


def to_numpy(x):
    """Convert to numpy array."""
    return x.numpy()


def to_ndarray(x, to_ndim, axis=0, dtype=None):
    """Convert to ndarray with specified dimensions."""
    x = _torch.as_tensor(x, dtype=dtype)
    if x.dim() > to_ndim:
        raise ValueError("The ndim cannot be adapted properly.")
    while x.dim() < to_ndim:
        x = _torch.unsqueeze(x, dim=axis)
    return x


def sort(a, axis=-1):
    """Sort array."""
    sorted_a, _ = _torch.sort(a, dim=axis)
    return sorted_a


def take(a, indices, axis=0):
    """Take elements from array."""
    if not _torch.is_tensor(indices):
        indices = _torch.as_tensor(indices)
    return _torch.squeeze(_torch.index_select(a, axis, indices))


def unique(ar, axis=None):
    """Return unique elements."""
    return _torch.unique(ar, dim=axis)


def hsplit(x, indices_or_section):
    """Split array horizontally."""
    if isinstance(indices_or_section, int):
        indices_or_section = x.shape[-1] // indices_or_section
    return _torch.split(x, indices_or_section, dim=-1)


def get_slice(x, indices):
    """Return a slice of an array."""
    return x[indices]


def scatter_add(input, dim, index, src):
    """Scatter add operation."""
    return _torch.scatter_add(input, dim, index, src)


# =============================================================================
# Specialized functions
# =============================================================================


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return _is_boolean(x[0])
    if _torch.is_tensor(x):
        return x.dtype in [_torch.bool, _torch.uint8]
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if _torch.is_tensor(x):
        return ndim(x) > 0
    return False


def assignment(x, values, indices, axis=0):
    """Assign values at given indices."""
    x_new = copy(x)
    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] = values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        x_new[indices] = values
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] = values
    return x_new


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices."""
    x_new = copy(x)
    values = array(values)
    use_vectorization = hasattr(indices, "__len__") and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] += values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        x_new[indices] += values
    else:
        indices = tuple(list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] += values
    return x_new


def one_hot(labels, num_classes):
    """One-hot encode labels."""
    if not _torch.is_tensor(labels):
        labels = _torch.LongTensor(labels)
    return _torch.nn.functional.one_hot(labels, num_classes).type(_torch.uint8)


def divide(a, b, ignore_div_zero=False):
    """Division with optional zero handling."""
    if not ignore_div_zero:
        return _torch.divide(a, b)
    quo = _torch.divide(a, b)
    return _torch.nan_to_num(quo, nan=0.0, posinf=0.0, neginf=0.0)


def set_diag(x, new_diag):
    """Set diagonal elements."""
    arr_shape = x.shape
    off_diag = (1 - _torch.eye(arr_shape[-1])) * x
    diag = _torch.einsum("ij,...i->...ij", _torch.eye(new_diag.shape[-1]), new_diag)
    return diag + off_diag


def tril_to_vec(x, k=0):
    """Extract lower triangle as vector."""
    n = x.shape[-1]
    rows, cols = tril_indices(n, k=k)
    return x[..., rows, cols]


def triu_to_vec(x, k=0):
    """Extract upper triangle as vector."""
    n = x.shape[-1]
    rows, cols = triu_indices(n, k=k)
    return x[..., rows, cols]


def mat_from_diag_triu_tril(diag, tri_upp, tri_low):
    """Build matrix from components."""
    diag, tri_upp, tri_low = convert_to_wider_dtype([diag, tri_upp, tri_low])
    n = diag.shape[-1]
    (i,) = diag_indices(n, ndim=1)
    j, k = triu_indices(n, k=1)
    mat = _torch.zeros((diag.shape + (n,)), dtype=diag.dtype)
    mat[..., i, i] = diag
    mat[..., j, k] = tri_upp
    mat[..., k, j] = tri_low
    return mat


def array_from_sparse(indices, data, target_shape):
    """Create array from sparse indices and data."""
    return _torch.sparse_coo_tensor(
        _torch.LongTensor(indices).t(),
        array(data),
        _torch.Size(target_shape),
    ).to_dense()


def ravel_tril_indices(n, k=0, m=None):
    """Return raveled indices for lower triangle."""
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = _np.tril_indices(n, k, m)
    return _torch.from_numpy(_np.ravel_multi_index(idxs, size))


def vectorize(x, pyfunc, multiple_args=False, **kwargs):
    """Vectorize a function."""
    if multiple_args:
        return stack(list(map(lambda y: pyfunc(*y), zip(*x))))
    return stack(list(map(pyfunc, x)))


def _unnest_iterable(ls):
    out = []
    if isinstance(ls, _Iterable):
        for inner_ls in ls:
            out.extend(_unnest_iterable(inner_ls))
    else:
        out.append(ls)
    return out


def pad(a, pad_width, mode="constant", constant_values=0.0):
    """Pad array."""
    return _torch.nn.functional.pad(
        a, _unnest_iterable(reversed(pad_width)), mode=mode, value=constant_values
    )
