"""VJPs registered for the functions autograd cannot differentiate."""

import functools as _functools

import autograd.numpy as _np
from autograd.extend import defvjp as _defvjp
from autograd.extend import defvjp_argnums as _defvjp_argnums
from autograd.numpy.numpy_vjps import unbroadcast_f as _unbroadcast_f
from autograd.scipy.linalg import expm as _expm

from .linalg import logm as _logm
from .linalg import solve_sylvester as _solve_sylvester


def _matvec_vjp_mat(ans, mat, vec):
    """Get the VJP of the matrix-vector product with respect to the matrix."""
    return _unbroadcast_f(mat, lambda g: g[..., :, None] * vec[..., None, :])


def _matvec_vjp_vec(ans, mat, vec):
    """Get the VJP of the matrix-vector product with respect to the vector."""
    return _unbroadcast_f(vec, lambda g: _np.matvec(_np.swapaxes(mat, -1, -2), g))


def _vecdot_vjp_a(ans, a, b):
    """Get the VJP of the vector dot product with respect to the first vector."""
    return _unbroadcast_f(a, lambda g: _np.conj(g[..., None] * b))


def _vecdot_vjp_b(ans, a, b):
    """Get the VJP of the vector dot product with respect to the second vector."""
    return _unbroadcast_f(b, lambda g: g[..., None] * _np.conj(a))


def _matrix_transpose_vjp(ans, x):
    """Get the VJP of the matrix transpose."""
    return lambda g: _np.matrix_transpose(g)


def _vector_norm_vjp(ans, x, axis=None, keepdims=False, ord=2):
    """Get the VJP of the vector norm, for a finite order greater than one."""
    if not 1 < ord < float("inf"):
        raise NotImplementedError(f"VJP of vector_norm not implemented for ord={ord}")

    def expand(a):
        return a if keepdims or axis is None else _np.expand_dims(a, axis)

    return lambda g: (
        expand(g) * _np.conj(x) * _np.abs(x) ** (ord - 2) / expand(ans) ** (ord - 1)
    )


def _matrix_norm_vjp(ans, x, keepdims=False, ord="fro"):
    """Get the VJP of the Frobenius matrix norm."""
    if ord != "fro":
        raise NotImplementedError(f"VJP of matrix_norm not implemented for ord={ord}")

    def expand(a):
        return a if keepdims else a[..., None, None]

    return lambda g: expand(g) * _np.conj(x) / expand(ans)


def _trace_vjp(ans, x):
    """Get the VJP of the trace of square matrices."""
    return lambda g: g[..., None, None] * _np.eye(x.shape[-1])


def _solve_sylvester_vjp(argnums, ans, args, kwargs):
    """Get the VJPs of the solution of the Sylvester equation.

    If ``X`` solves ``A X + X B = Q`` and ``Y`` solves ``A^T Y + Y B^T = g``,
    the VJPs with respect to ``A``, ``B`` and ``Q`` are ``-Y X^T``, ``-X^T Y``
    and ``Y``. Unlike differentiating through the solver, this does not
    involve the eigenvectors of ``A``, whose gradient is undefined for
    repeated eigenvalues.
    """
    a, b, _ = args

    def vjp(g):
        y = _solve_sylvester(
            _np.matrix_transpose(a), _np.matrix_transpose(b), g, **kwargs
        )
        transposed_x = _np.matrix_transpose(ans)
        grads = (-y @ transposed_x, -transposed_x @ y, y)
        return tuple(
            _unbroadcast_f(args[argnum], lambda grad: grad)(grads[argnum])
            for argnum in argnums
        )

    return vjp


def _matrix_function_vjp(_ans, x, fn):
    """Get the VJP of a matrix function.

    The VJP is the upper right block of ``fn([[x^T, g], [0, x^T]])``. The block
    matrix is built with differentiable operations, so that the VJP can itself
    be differentiated.
    """
    transposed_x = _np.matrix_transpose(x)
    zeros = _np.zeros_like(transposed_x)
    n = x.shape[-1]

    def vjp(g):
        mat = _np.concatenate(
            [
                _np.concatenate([transposed_x, g], axis=-1),
                _np.concatenate([zeros, transposed_x], axis=-1),
            ],
            axis=-2,
        )
        return fn(mat)[..., :n, n:]

    return vjp


_defvjp(_np.matrix_transpose, _matrix_transpose_vjp)
_defvjp(_np.matvec, _matvec_vjp_mat, _matvec_vjp_vec)
_defvjp(_np.linalg.vecdot, _vecdot_vjp_a, _vecdot_vjp_b)
_defvjp(_np.linalg.vector_norm, _vector_norm_vjp)
_defvjp(_np.linalg.matrix_norm, _matrix_norm_vjp)
_defvjp(_np.linalg.trace, _trace_vjp)
_defvjp(_expm, _functools.partial(_matrix_function_vjp, fn=_expm))
_defvjp(_logm, _functools.partial(_matrix_function_vjp, fn=_logm))
_defvjp_argnums(_solve_sylvester, _solve_sylvester_vjp)
