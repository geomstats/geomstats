"""Shared utilities for numpy-based backends (numpy and autograd)."""

from ._dispatch import BACKEND_NAME, _common
from ._dispatch import numpy as _np

_is_iterable = _common._is_iterable
_is_boolean = _common._is_boolean
_get_wider_dtype = _common._get_wider_dtype
array = _common.array
cast = _common.cast
convert_to_wider_dtype = _common.convert_to_wider_dtype
eye = _common.eye
is_array = _common.is_array
get_default_dtype = _common.get_default_dtype
zeros = _common.zeros
_box_binary_scalar = _common._box_binary_scalar

mod = _box_binary_scalar(target=_np.mod)


def angle(z, deg=False):
    """Return the angle of the complex argument."""
    out = _np.angle(z, deg=deg)
    if isinstance(z, float):
        return cast(out, get_default_dtype())
    return out


def arange(start_or_stop, /, stop=None, step=1, dtype=None, **kwargs):
    """Return evenly spaced values within a given interval."""
    if dtype is None and (
        isinstance(stop, float)
        or isinstance(step, float)
        or isinstance(start_or_stop, float)
    ):
        dtype = get_default_dtype()
    if stop is None:
        return _np.arange(start_or_stop, step=step, dtype=dtype)
    return _np.arange(start_or_stop, stop, step=step, dtype=dtype)


def to_numpy(x):
    """Convert to numpy array (no-op for numpy backend)."""
    return x


def from_numpy(x):
    """Convert from numpy array (no-op for numpy backend)."""
    return x


def flatten(x):
    """Flatten array to 1D."""
    return x.flatten()


def copy(x):
    """Copy array."""
    return x.copy()


def ndim(x):
    """Return number of dimensions."""
    return x.ndim


def one_hot(labels, num_classes):
    """One-hot encode labels."""
    return eye(num_classes, dtype=_np.dtype("uint8"))[labels]


def get_slice(x, indices):
    """Return a slice of an array."""
    return x[indices]


def assignment(x, values, indices, axis=0):
    """Assign values at given indices of an array."""
    x_new = x.copy()
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
    """Add values at given indices of an array."""
    x_new = x.copy()
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


def set_diag(x, new_diag):
    """Set the diagonal along the last two axes."""
    arr_shape = x.shape
    x[..., range(arr_shape[-2]), range(arr_shape[-1])] = new_diag
    return x


def array_from_sparse(indices, data, target_shape):
    """Create array from sparse indices and data."""
    data = array(data)
    out = zeros(target_shape, dtype=data.dtype)
    out.put(_np.ravel_multi_index(_np.array(indices).T, target_shape), data)
    return out


def vec_to_diag(vec):
    """Convert vector to diagonal matrix."""
    d = vec.shape[-1]
    return _np.squeeze(vec[..., None, :] * eye(d, dtype=vec.dtype)[None, :, :])


def tril_to_vec(x, k=0):
    """Extract lower triangle as vector."""
    n = x.shape[-1]
    rows, cols = _np.tril_indices(n, k=k)
    return x[..., rows, cols]


def triu_to_vec(x, k=0):
    """Extract upper triangle as vector."""
    n = x.shape[-1]
    rows, cols = _np.triu_indices(n, k=k)
    return x[..., rows, cols]


def mat_from_diag_triu_tril(diag, tri_upp, tri_low):
    """Build matrix from diagonal, upper tri, and lower tri components."""
    diag, tri_upp, tri_low = convert_to_wider_dtype([diag, tri_upp, tri_low])
    n = diag.shape[-1]
    (i,) = _np.diag_indices(n, ndim=1)
    j, k = _np.triu_indices(n, k=1)
    mat = zeros(diag.shape + (n,), dtype=diag.dtype)
    mat[..., i, i] = diag
    mat[..., j, k] = tri_upp
    mat[..., k, j] = tri_low
    return mat


def divide(a, b, ignore_div_zero=False):
    """Division with optional zero handling."""
    if not ignore_div_zero:
        return _np.divide(a, b)
    wider_dtype, _ = _get_wider_dtype([a, b])
    return _np.divide(a, b, out=zeros(a.shape, dtype=wider_dtype), where=b != 0)


def ravel_tril_indices(n, k=0, m=None):
    """Return raveled indices for lower triangle."""
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = _np.tril_indices(n, k, m)
    return _np.ravel_multi_index(idxs, size)


def matvec(A, b):
    """Matrix-vector product."""
    if b.ndim == 1:
        return _np.matmul(A, b)
    if A.ndim == 2:
        return _np.matmul(A, b.T).T
    return _np.einsum("...ij,...j->...i", A, b)


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    """Vectorize a function."""
    if multiple_args:
        return _np.vectorize(pyfunc, signature=signature)(*x)
    return _np.vectorize(pyfunc, signature=signature)(x)


def scatter_add(input, dim, index, src):
    """Add values from src into input at specified indices."""
    if dim == 0:
        for i, val in zip(index, src):
            input[i] += val
        return input
    if dim == 1:
        for j in range(len(input)):
            for i, val in zip(index[j], src[j]):
                if not isinstance(val, _np.float64) and BACKEND_NAME == "autograd":
                    val = float(val._value)
                input[j, i] += float(val)
        return input
    raise NotImplementedError
