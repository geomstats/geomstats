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
