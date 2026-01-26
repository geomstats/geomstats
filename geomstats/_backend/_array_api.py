"""Array API standard utilities for backend-agnostic operations.

This module provides helpers to use Python Array API standard functions
that work identically across numpy, pytorch, and other compatible backends.

The Array API standard (https://data-apis.org/array-api/) provides a common
interface for array operations across different libraries. NumPy 2.0+ and
PyTorch 2.0+ both support this standard.
"""

import numpy as np

# For PyTorch compatibility with Array API
try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# For autograd compatibility
try:
    import autograd.numpy as anp

    _HAS_AUTOGRAD = True
except ImportError:
    _HAS_AUTOGRAD = False


def get_namespace(*arrays):
    """Get the array namespace (xp) for the given arrays.

    Uses numpy 2.0's __array_namespace__ protocol.
    Falls back to numpy for Python scalars/lists.

    Parameters
    ----------
    arrays : array-like
        Input arrays to determine namespace from.

    Returns
    -------
    xp : module
        Array namespace (numpy, torch, etc.)
    """
    for arr in arrays:
        # Check for Array API namespace protocol
        if hasattr(arr, "__array_namespace__"):
            return arr.__array_namespace__()
        # PyTorch tensors
        if _HAS_TORCH and isinstance(arr, torch.Tensor):
            return torch
        # Autograd arrays (wrapped numpy arrays)
        if _HAS_AUTOGRAD and hasattr(arr, "_value"):
            # Autograd ArrayBox wraps values
            return anp
    return np


def is_torch_tensor(x):
    """Check if x is a PyTorch tensor.

    Parameters
    ----------
    x : array-like
        Input to check.

    Returns
    -------
    bool
        True if x is a PyTorch tensor.
    """
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def is_autograd_array(x):
    """Check if x is an autograd array.

    Parameters
    ----------
    x : array-like
        Input to check.

    Returns
    -------
    bool
        True if x is an autograd ArrayBox.
    """
    return _HAS_AUTOGRAD and hasattr(x, "_value")


def is_array_api_compatible(x):
    """Check if x supports the Array API standard.

    Parameters
    ----------
    x : array-like
        Input to check.

    Returns
    -------
    bool
        True if x has __array_namespace__ method.
    """
    return hasattr(x, "__array_namespace__")


# =============================================================================
# Helper for scalar handling
# =============================================================================


def _ensure_array(x):
    """Convert Python scalar to array if needed.

    Parameters
    ----------
    x : scalar or array-like
        Input value.

    Returns
    -------
    array-like
        Array version of x.
    """
    if isinstance(x, (int, float, complex)):
        return np.asarray(x)
    return x


# =============================================================================
# Math Functions (Array API standard)
# =============================================================================


def abs(x):  # noqa: A001
    """Absolute value, element-wise."""
    xp = get_namespace(x)
    return xp.abs(x)


def sin(x):
    """Trigonometric sine, element-wise."""
    xp = get_namespace(x)
    return xp.sin(x)


def cos(x):
    """Trigonometric cosine, element-wise."""
    xp = get_namespace(x)
    return xp.cos(x)


def tan(x):
    """Trigonometric tangent, element-wise."""
    xp = get_namespace(x)
    return xp.tan(x)


def arcsin(x):
    """Inverse sine, element-wise.

    Note: Array API uses 'asin', numpy uses 'arcsin'.
    """
    xp = get_namespace(x)
    if hasattr(xp, "arcsin"):
        return xp.arcsin(x)
    return xp.asin(x)


def arccos(x):
    """Inverse cosine, element-wise.

    Note: Array API uses 'acos', numpy uses 'arccos'.
    """
    xp = get_namespace(x)
    if hasattr(xp, "arccos"):
        return xp.arccos(x)
    return xp.acos(x)


def arctan2(y, x):
    """Element-wise arc tangent of y/x.

    Note: Array API uses 'atan2', numpy uses 'arctan2'.
    """
    xp = get_namespace(x, y)
    if hasattr(xp, "arctan2"):
        return xp.arctan2(y, x)
    return xp.atan2(y, x)


def sinh(x):
    """Hyperbolic sine, element-wise."""
    xp = get_namespace(x)
    return xp.sinh(x)


def cosh(x):
    """Hyperbolic cosine, element-wise."""
    xp = get_namespace(x)
    return xp.cosh(x)


def tanh(x):
    """Hyperbolic tangent, element-wise."""
    xp = get_namespace(x)
    return xp.tanh(x)


def arccosh(x):
    """Inverse hyperbolic cosine, element-wise.

    Note: Array API uses 'acosh', numpy uses 'arccosh'.
    """
    xp = get_namespace(x)
    if hasattr(xp, "arccosh"):
        return xp.arccosh(x)
    return xp.acosh(x)


def arctanh(x):
    """Inverse hyperbolic tangent, element-wise.

    Note: Array API uses 'atanh', numpy uses 'arctanh'.
    """
    xp = get_namespace(x)
    if hasattr(xp, "arctanh"):
        return xp.arctanh(x)
    return xp.atanh(x)


def exp(x):
    """Exponential, element-wise."""
    xp = get_namespace(x)
    return xp.exp(x)


def log(x):
    """Natural logarithm, element-wise."""
    xp = get_namespace(x)
    return xp.log(x)


def sqrt(x):
    """Square root, element-wise."""
    xp = get_namespace(x)
    return xp.sqrt(x)


def ceil(x):
    """Ceiling, element-wise."""
    xp = get_namespace(x)
    return xp.ceil(x)


def floor(x):
    """Floor, element-wise."""
    xp = get_namespace(x)
    return xp.floor(x)


def sign(x):
    """Sign of elements."""
    xp = get_namespace(x)
    return xp.sign(x)


def real(x):
    """Real part of complex argument."""
    xp = get_namespace(x)
    return xp.real(x)


def imag(x):
    """Imaginary part of complex argument."""
    xp = get_namespace(x)
    return xp.imag(x)


def conj(x):
    """Complex conjugate, element-wise."""
    xp = get_namespace(x)
    return xp.conj(x)


def power(x1, x2):
    """First array elements raised to powers from second array."""
    xp = get_namespace(x1, x2)
    # NumPy uses 'power', torch uses 'pow'
    if hasattr(xp, "power"):
        return xp.power(x1, x2)
    return xp.pow(x1, x2)


# =============================================================================
# Array Creation Functions (Array API standard)
# =============================================================================


def zeros_like(x, dtype=None):
    """Return an array of zeros with the same shape and type as a given array.

    Parameters
    ----------
    x : array-like
        The shape and dtype of x define these same attributes of the returned array.
    dtype : dtype, optional
        Overrides the dtype of the result.

    Returns
    -------
    out : ndarray
        Array of zeros with the same shape as x.
    """
    xp = get_namespace(x)
    return xp.zeros_like(x, dtype=dtype)


def ones_like(x, dtype=None):
    """Return an array of ones with the same shape and type as a given array.

    Parameters
    ----------
    x : array-like
        The shape and dtype of x define these same attributes of the returned array.
    dtype : dtype, optional
        Overrides the dtype of the result.

    Returns
    -------
    out : ndarray
        Array of ones with the same shape as x.
    """
    xp = get_namespace(x)
    return xp.ones_like(x, dtype=dtype)


def empty_like(x, dtype=None):
    """Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    x : array-like
        The shape and dtype of x define these same attributes of the returned array.
    dtype : dtype, optional
        Overrides the dtype of the result.

    Returns
    -------
    out : ndarray
        Array with the same shape as x.
    """
    xp = get_namespace(x)
    return xp.empty_like(x, dtype=dtype)


# =============================================================================
# Array Manipulation Functions
# =============================================================================


def reshape(x, shape):
    """Reshape an array.

    Parameters
    ----------
    x : array-like
        Array to reshape.
    shape : tuple of ints
        New shape.

    Returns
    -------
    out : array-like
        Reshaped array.
    """
    xp = get_namespace(x)
    return xp.reshape(x, shape)


def transpose(x, axes=None):
    """Permute the dimensions of an array.

    Parameters
    ----------
    x : array-like
        Input array.
    axes : tuple of ints, optional
        Permutation of dimensions.

    Returns
    -------
    out : array-like
        Transposed array.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axes is None:
            return x.T if x.ndim <= 2 else x.permute(tuple(range(x.ndim)[::-1]))
        return x.permute(axes)
    return xp.transpose(x, axes)


def squeeze(x, axis=None):
    """Remove axes of length one from array.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int or tuple of ints, optional
        Axes to squeeze.

    Returns
    -------
    out : array-like
        Squeezed array.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.squeeze()
        return x.squeeze(dim=axis)
    if axis is None:
        return xp.squeeze(x)
    # Handle case where axis dimension is not 1
    if hasattr(x, "shape") and x.shape[axis] != 1:
        return x
    return xp.squeeze(x, axis=axis)


def expand_dims(x, axis):
    """Expand the shape of an array by inserting a new axis.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int
        Position where new axis is inserted.

    Returns
    -------
    out : array-like
        Array with expanded shape.
    """
    xp = get_namespace(x)
    if xp is torch:
        return x.unsqueeze(dim=axis)
    return xp.expand_dims(x, axis=axis)


def concatenate(arrays, axis=0):
    """Join arrays along an existing axis.

    Parameters
    ----------
    arrays : sequence of array-like
        Arrays to concatenate.
    axis : int, optional
        Axis along which to concatenate.

    Returns
    -------
    out : array-like
        Concatenated array.
    """
    xp = get_namespace(*arrays)
    if xp is torch:
        return torch.cat(arrays, dim=axis)
    return xp.concatenate(arrays, axis=axis)


def stack(arrays, axis=0):
    """Join arrays along a new axis.

    Parameters
    ----------
    arrays : sequence of array-like
        Arrays to stack.
    axis : int, optional
        Axis along which to stack.

    Returns
    -------
    out : array-like
        Stacked array.
    """
    xp = get_namespace(*arrays)
    if xp is torch:
        return torch.stack(arrays, dim=axis)
    return xp.stack(arrays, axis=axis)


def split(x, indices_or_sections, axis=0):
    """Split array into multiple sub-arrays.

    Parameters
    ----------
    x : array-like
        Array to split.
    indices_or_sections : int or sequence of ints
        Number of sections or indices where to split.
    axis : int, optional
        Axis along which to split.

    Returns
    -------
    out : list of array-like
        List of sub-arrays.
    """
    xp = get_namespace(x)
    if xp is torch:
        if isinstance(indices_or_sections, int):
            split_size = x.shape[axis] // indices_or_sections
            return torch.split(x, split_size, dim=axis)
        # Convert indices to sizes for torch.split
        indices = list(indices_or_sections)
        sizes = []
        prev = 0
        for idx in indices:
            sizes.append(idx - prev)
            prev = idx
        sizes.append(x.shape[axis] - prev)
        return torch.split(x, sizes, dim=axis)
    return xp.split(x, indices_or_sections, axis=axis)


def flip(x, axis=None):
    """Reverse elements along given axis.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes to flip.

    Returns
    -------
    out : array-like
        Flipped array.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            axis = list(range(x.ndim))
        elif isinstance(axis, int):
            axis = [axis]
        return torch.flip(x, dims=axis)
    return xp.flip(x, axis=axis)


def tile(x, reps):
    """Repeat array along each axis.

    Parameters
    ----------
    x : array-like
        Input array.
    reps : tuple of ints
        Number of repetitions along each axis.

    Returns
    -------
    out : array-like
        Tiled array.
    """
    xp = get_namespace(x)
    if xp is torch:
        return x.repeat(reps)
    return xp.tile(x, reps)


def moveaxis(x, source, destination):
    """Move axes to new positions.

    Parameters
    ----------
    x : array-like
        Input array.
    source : int or sequence of ints
        Original positions of axes to move.
    destination : int or sequence of ints
        Destination positions.

    Returns
    -------
    out : array-like
        Array with moved axes.
    """
    xp = get_namespace(x)
    return xp.moveaxis(x, source, destination)


def broadcast_to(x, shape):
    """Broadcast array to a new shape.

    Parameters
    ----------
    x : array-like
        Input array.
    shape : tuple of ints
        Target shape.

    Returns
    -------
    out : array-like
        Broadcasted array.
    """
    xp = get_namespace(x)
    if xp is torch:
        return x.expand(shape)
    return xp.broadcast_to(x, shape)


def diagonal(x, offset=0, axis1=0, axis2=1):
    """Return specified diagonals.

    Parameters
    ----------
    x : array-like
        Input array.
    offset : int, optional
        Offset from main diagonal.
    axis1, axis2 : int, optional
        Axes to use for 2D sub-arrays.

    Returns
    -------
    out : array-like
        Diagonal elements.
    """
    xp = get_namespace(x)
    if xp is torch:
        return torch.diagonal(x, offset=offset, dim1=axis1, dim2=axis2)
    return xp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def tril(x, k=0):
    """Lower triangle of an array.

    Parameters
    ----------
    x : array-like
        Input array.
    k : int, optional
        Diagonal offset.

    Returns
    -------
    out : array-like
        Lower triangular array.
    """
    xp = get_namespace(x)
    if xp is torch:
        return torch.tril(x, diagonal=k)
    return xp.tril(x, k=k)


def triu(x, k=0):
    """Upper triangle of an array.

    Parameters
    ----------
    x : array-like
        Input array.
    k : int, optional
        Diagonal offset.

    Returns
    -------
    out : array-like
        Upper triangular array.
    """
    xp = get_namespace(x)
    if xp is torch:
        return torch.triu(x, diagonal=k)
    return xp.triu(x, k=k)


def repeat(x, repeats, axis=None):
    """Repeat elements of an array.

    Parameters
    ----------
    x : array-like
        Input array.
    repeats : int or sequence of ints
        Number of repetitions.
    axis : int, optional
        Axis along which to repeat.

    Returns
    -------
    out : array-like
        Array with repeated elements.
    """
    xp = get_namespace(x)
    if xp is torch:
        return torch.repeat_interleave(x, repeats, dim=axis)
    return xp.repeat(x, repeats, axis=axis)


def hstack(arrays):
    """Stack arrays horizontally.

    Parameters
    ----------
    arrays : sequence of array-like
        Arrays to stack.

    Returns
    -------
    out : array-like
        Stacked array.
    """
    xp = get_namespace(*arrays)
    return xp.hstack(arrays)


def vstack(arrays):
    """Stack arrays vertically.

    Parameters
    ----------
    arrays : sequence of array-like
        Arrays to stack.

    Returns
    -------
    out : array-like
        Stacked array.
    """
    xp = get_namespace(*arrays)
    return xp.vstack(arrays)


# =============================================================================
# Reduction Operations
# =============================================================================


def sum(x, axis=None, keepdims=False, dtype=None):  # noqa: A001
    """Sum of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes to sum over.
    keepdims : bool, optional
        Keep reduced dimensions.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    out : array-like
        Sum of elements.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.sum(dtype=dtype)
        return x.sum(dim=axis, keepdim=keepdims, dtype=dtype)
    return xp.sum(x, axis=axis, keepdims=keepdims, dtype=dtype)


def mean(x, axis=None, keepdims=False, dtype=None):
    """Mean of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes to compute mean over.
    keepdims : bool, optional
        Keep reduced dimensions.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    out : array-like
        Mean of elements.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.mean(dtype=dtype)
        return x.mean(dim=axis, keepdim=keepdims, dtype=dtype)
    return xp.mean(x, axis=axis, keepdims=keepdims, dtype=dtype)


def prod(x, axis=None, keepdims=False, dtype=None):
    """Product of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis to compute product over.
    keepdims : bool, optional
        Keep reduced dimensions.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    out : array-like
        Product of elements.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.prod(dtype=dtype)
        return x.prod(dim=axis, keepdim=keepdims, dtype=dtype)
    return xp.prod(x, axis=axis, keepdims=keepdims, dtype=dtype)


def std(x, axis=None, keepdims=False, ddof=0):
    """Standard deviation of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes to compute std over.
    keepdims : bool, optional
        Keep reduced dimensions.
    ddof : int, optional
        Delta degrees of freedom.

    Returns
    -------
    out : array-like
        Standard deviation.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.std(unbiased=(ddof == 1))
        return x.std(dim=axis, keepdim=keepdims, unbiased=(ddof == 1))
    return xp.std(x, axis=axis, keepdims=keepdims, ddof=ddof)


def amax(x, axis=None, keepdims=False):
    """Maximum of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis to compute max over.
    keepdims : bool, optional
        Keep reduced dimensions.

    Returns
    -------
    out : array-like
        Maximum value.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.max()
        return x.max(dim=axis, keepdim=keepdims).values
    return xp.amax(x, axis=axis, keepdims=keepdims)


def amin(x, axis=None, keepdims=False):
    """Minimum of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis to compute min over.
    keepdims : bool, optional
        Keep reduced dimensions.

    Returns
    -------
    out : array-like
        Minimum value.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.min()
        return x.min(dim=axis, keepdim=keepdims).values
    return xp.amin(x, axis=axis, keepdims=keepdims)


def argmax(x, axis=None, keepdims=False):
    """Indices of maximum values.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis to find max over.
    keepdims : bool, optional
        Keep reduced dimensions.

    Returns
    -------
    out : array-like
        Indices of max values.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.argmax()
        return x.argmax(dim=axis, keepdim=keepdims)
    return xp.argmax(x, axis=axis, keepdims=keepdims)


def argmin(x, axis=None, keepdims=False):
    """Indices of minimum values.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis to find min over.
    keepdims : bool, optional
        Keep reduced dimensions.

    Returns
    -------
    out : array-like
        Indices of min values.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.argmin()
        return x.argmin(dim=axis, keepdim=keepdims)
    return xp.argmin(x, axis=axis, keepdims=keepdims)


def all(x, axis=None, keepdims=False):  # noqa: A001
    """Test if all elements are true.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes to test over.
    keepdims : bool, optional
        Keep reduced dimensions.

    Returns
    -------
    out : array-like
        Boolean result.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.all()
        return x.all(dim=axis, keepdim=keepdims)
    return xp.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):  # noqa: A001
    """Test if any element is true.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int or tuple of ints, optional
        Axis or axes to test over.
    keepdims : bool, optional
        Keep reduced dimensions.

    Returns
    -------
    out : array-like
        Boolean result.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.any()
        return x.any(dim=axis, keepdim=keepdims)
    return xp.any(x, axis=axis, keepdims=keepdims)


def cumsum(x, axis=None, dtype=None):
    """Cumulative sum of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis to compute cumsum over.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    out : array-like
        Cumulative sum.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            return x.flatten().cumsum(dim=0, dtype=dtype)
        return x.cumsum(dim=axis, dtype=dtype)
    return xp.cumsum(x, axis=axis, dtype=dtype)


def cumprod(x, axis=None, dtype=None):
    """Cumulative product of array elements.

    Parameters
    ----------
    x : array-like
        Input array.
    axis : int, optional
        Axis to compute cumprod over.
    dtype : dtype, optional
        Output dtype.

    Returns
    -------
    out : array-like
        Cumulative product.
    """
    xp = get_namespace(x)
    if xp is torch:
        if axis is None:
            axis = 0
        return x.cumprod(dim=axis, dtype=dtype)
    return xp.cumprod(x, axis=axis, dtype=dtype)


# =============================================================================
# Comparison and Logical Operations
# =============================================================================


def equal(x1, x2):
    """Element-wise equality comparison.

    Parameters
    ----------
    x1, x2 : array-like
        Arrays to compare.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(x1, x2)
    return xp.equal(x1, x2)


def greater(x1, x2):
    """Element-wise greater-than comparison.

    Parameters
    ----------
    x1, x2 : array-like
        Arrays to compare.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(x1, x2)
    return xp.greater(x1, x2)


def less(x1, x2):
    """Element-wise less-than comparison.

    Parameters
    ----------
    x1, x2 : array-like
        Arrays to compare.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(x1, x2)
    return xp.less(x1, x2)


def less_equal(x1, x2):
    """Element-wise less-than-or-equal comparison.

    Parameters
    ----------
    x1, x2 : array-like
        Arrays to compare.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(x1, x2)
    if xp is torch:
        return torch.le(x1, x2)
    return xp.less_equal(x1, x2)


def logical_and(x1, x2):
    """Element-wise logical AND.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(x1, x2)
    return xp.logical_and(x1, x2)


def logical_or(x1, x2):
    """Element-wise logical OR.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(x1, x2)
    return xp.logical_or(x1, x2)


def isnan(x):
    """Element-wise test for NaN.

    Parameters
    ----------
    x : array-like
        Input array.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(x)
    return xp.isnan(x)


def isclose(a, b, rtol=1e-5, atol=1e-8):
    """Element-wise test for approximate equality.

    Parameters
    ----------
    a, b : array-like
        Input arrays.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    out : array-like
        Boolean array.
    """
    xp = get_namespace(a, b)
    return xp.isclose(a, b, rtol=rtol, atol=atol)


def allclose(a, b, rtol=1e-5, atol=1e-8):
    """Test if all elements are approximately equal.

    Parameters
    ----------
    a, b : array-like
        Input arrays.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.

    Returns
    -------
    out : bool
        True if all elements are close.
    """
    xp = get_namespace(a, b)
    return xp.allclose(a, b, rtol=rtol, atol=atol)


def where(condition, x=None, y=None):
    """Return elements chosen from x or y depending on condition.

    Parameters
    ----------
    condition : array-like
        Boolean condition.
    x, y : array-like, optional
        Values to choose from.

    Returns
    -------
    out : array-like
        Array of selected values.
    """
    xp = get_namespace(condition)
    if x is None and y is None:
        return xp.where(condition)
    return xp.where(condition, x, y)


def maximum(x1, x2):
    """Element-wise maximum of array elements.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Element-wise maximum.
    """
    xp = get_namespace(x1, x2)
    if xp is torch:
        # Convert scalars to tensors for torch.maximum
        if not torch.is_tensor(x1):
            x1 = torch.tensor(x1)
        if not torch.is_tensor(x2):
            x2 = torch.tensor(x2)
        return torch.maximum(x1, x2)
    return xp.maximum(x1, x2)


def minimum(x1, x2):
    """Element-wise minimum of array elements.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Element-wise minimum.
    """
    xp = get_namespace(x1, x2)
    if xp is torch:
        # Convert scalars to tensors for torch.minimum
        if not torch.is_tensor(x1):
            x1 = torch.tensor(x1)
        if not torch.is_tensor(x2):
            x2 = torch.tensor(x2)
        return torch.minimum(x1, x2)
    return xp.minimum(x1, x2)


def clip(x, a_min=None, a_max=None):
    """Clip array values to a range.

    Parameters
    ----------
    x : array-like
        Input array.
    a_min : scalar, optional
        Minimum value.
    a_max : scalar, optional
        Maximum value.

    Returns
    -------
    out : array-like
        Clipped array.
    """
    xp = get_namespace(x)
    if xp is torch:
        return torch.clamp(x, min=a_min, max=a_max)
    return xp.clip(x, a_min, a_max)


# =============================================================================
# Linear Algebra Operations
# =============================================================================


def matmul(x1, x2):
    """Matrix multiplication.

    Parameters
    ----------
    x1, x2 : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Matrix product.
    """
    xp = get_namespace(x1, x2)
    return xp.matmul(x1, x2)


def dot(a, b):
    """Dot product of two arrays.

    For 1D arrays, returns the inner product.
    For higher dimensions, returns element-wise dot product over last axis.

    Parameters
    ----------
    a, b : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Dot product.
    """
    xp = get_namespace(a, b)
    # Use einsum for all cases - works across backends
    if hasattr(a, "ndim") and hasattr(b, "ndim"):
        if b.ndim == 1:
            return xp.einsum("...i,i->...", a, b)
        if a.ndim == 1:
            return xp.einsum("i,...i->...", a, b)
        # Batched dot product over last axis
        return xp.einsum("...i,...i->...", a, b)
    return xp.einsum("...i,...i->...", a, b)


def einsum(subscripts, *operands):
    """Einstein summation convention.

    Parameters
    ----------
    subscripts : str
        Subscripts for summation.
    operands : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Result of einsum.
    """
    xp = get_namespace(*operands)
    return xp.einsum(subscripts, *operands)


def trace(x, offset=0, axis1=-2, axis2=-1):
    """Sum along diagonals.

    Parameters
    ----------
    x : array-like
        Input array.
    offset : int, optional
        Offset from main diagonal.
    axis1, axis2 : int, optional
        Axes to use for 2D sub-arrays.

    Returns
    -------
    out : array-like
        Sum along diagonals.
    """
    xp = get_namespace(x)
    # Use einsum for main diagonal (offset=0) - works across backends
    if offset == 0 and axis1 == -2 and axis2 == -1:
        return xp.einsum("...ii->...", x)
    # Fallback for other cases
    if xp is torch:
        if x.ndim == 2:
            return torch.trace(x)
        return torch.einsum("...ii->...", x)
    return xp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def cross(a, b, axis=-1):
    """Cross product of two vectors.

    Parameters
    ----------
    a, b : array-like
        Input arrays.
    axis : int, optional
        Axis along which to compute cross product.

    Returns
    -------
    out : array-like
        Cross product.
    """
    xp = get_namespace(a, b)
    if xp is torch:
        return torch.cross(a, b, dim=axis)
    return xp.cross(a, b, axis=axis)


def outer(a, b):
    """Outer product of two arrays.

    Parameters
    ----------
    a, b : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Outer product.
    """
    xp = get_namespace(a, b)
    # Use einsum for all cases - works across backends
    return xp.einsum("...i,...j->...ij", a, b)


def kron(a, b):
    """Kronecker product of two arrays.

    Parameters
    ----------
    a, b : array-like
        Input arrays.

    Returns
    -------
    out : array-like
        Kronecker product.
    """
    xp = get_namespace(a, b)
    return xp.kron(a, b)


# =============================================================================
# Array Creation Operations (shape-based)
# =============================================================================
# Note: These functions create new arrays and need a reference array or
# explicit backend specification. For backend-independent code, prefer
# using xxx_like functions or pass arrays to infer backend.


def zeros(shape, dtype=None, *, like=None):
    """Return array of zeros.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array.
    dtype : dtype, optional
        Data type.
    like : array-like, optional
        Reference array to determine backend.

    Returns
    -------
    out : array-like
        Array of zeros.
    """
    if like is not None:
        xp = get_namespace(like)
    else:
        xp = np
    if xp is torch:
        return torch.zeros(shape, dtype=dtype)
    return xp.zeros(shape, dtype=dtype)


def ones(shape, dtype=None, *, like=None):
    """Return array of ones.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array.
    dtype : dtype, optional
        Data type.
    like : array-like, optional
        Reference array to determine backend.

    Returns
    -------
    out : array-like
        Array of ones.
    """
    if like is not None:
        xp = get_namespace(like)
    else:
        xp = np
    if xp is torch:
        return torch.ones(shape, dtype=dtype)
    return xp.ones(shape, dtype=dtype)


def eye(n, m=None, k=0, dtype=None, *, like=None):
    """Return identity matrix.

    Parameters
    ----------
    n : int
        Number of rows.
    m : int, optional
        Number of columns.
    k : int, optional
        Diagonal offset.
    dtype : dtype, optional
        Data type.
    like : array-like, optional
        Reference array to determine backend.

    Returns
    -------
    out : array-like
        Identity matrix.
    """
    if like is not None:
        xp = get_namespace(like)
    else:
        xp = np
    if m is None:
        m = n
    if xp is torch:
        if k == 0:
            return torch.eye(n, m, dtype=dtype)
        # Torch doesn't support diagonal offset in eye
        result = torch.zeros(n, m, dtype=dtype)
        if k >= 0:
            for i in range(min(n, m - k)):
                result[i, i + k] = 1
        else:
            for i in range(min(n + k, m)):
                result[i - k, i] = 1
        return result
    return xp.eye(n, m, k=k, dtype=dtype)


def arange(start, stop=None, step=1, dtype=None, *, like=None):
    """Return evenly spaced values within a given interval.

    Parameters
    ----------
    start : number
        Start of interval.
    stop : number, optional
        End of interval.
    step : number, optional
        Step size.
    dtype : dtype, optional
        Data type.
    like : array-like, optional
        Reference array to determine backend.

    Returns
    -------
    out : array-like
        Array of evenly spaced values.
    """
    if like is not None:
        xp = get_namespace(like)
    else:
        xp = np
    if stop is None:
        stop = start
        start = 0
    if xp is torch:
        return torch.arange(start, stop, step, dtype=dtype)
    return xp.arange(start, stop, step, dtype=dtype)


def linspace(start, stop, num=50, endpoint=True, dtype=None, *, like=None):
    """Return evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start : number
        Start of interval.
    stop : number
        End of interval.
    num : int, optional
        Number of samples.
    endpoint : bool, optional
        Whether to include stop.
    dtype : dtype, optional
        Data type.
    like : array-like, optional
        Reference array to determine backend.

    Returns
    -------
    out : array-like
        Array of evenly spaced values.
    """
    if like is not None:
        xp = get_namespace(like)
    else:
        xp = np
    if xp is torch:
        if endpoint:
            return torch.linspace(start, stop, num, dtype=dtype)
        # Compute without endpoint
        step = (stop - start) / num
        return torch.arange(start, stop, step, dtype=dtype)[:num]
    return xp.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)


def full(shape, fill_value, dtype=None, *, like=None):
    """Return array filled with a scalar value.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the array.
    fill_value : scalar
        Fill value.
    dtype : dtype, optional
        Data type.
    like : array-like, optional
        Reference array to determine backend.

    Returns
    -------
    out : array-like
        Array filled with fill_value.
    """
    if like is not None:
        xp = get_namespace(like)
    else:
        xp = np
    if xp is torch:
        return torch.full(shape, fill_value, dtype=dtype)
    return xp.full(shape, fill_value, dtype=dtype)


def full_like(x, fill_value, dtype=None):
    """Return array filled with a scalar value, same shape as input.

    Parameters
    ----------
    x : array-like
        Reference array.
    fill_value : scalar
        Fill value.
    dtype : dtype, optional
        Data type.

    Returns
    -------
    out : array-like
        Array filled with fill_value.
    """
    xp = get_namespace(x)
    if xp is torch:
        return torch.full_like(x, fill_value, dtype=dtype)
    return xp.full_like(x, fill_value, dtype=dtype)
