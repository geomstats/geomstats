"""Numpy based linear algebra backend."""

import jax.numpy as np
import numpy as _np
import scipy.linalg
from jax import core, vmap, custom_vjp
from jax.numpy.linalg import (  # NOQA
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    norm,
    svd
)
from jax.scipy.linalg import expm as sp_expm

from .common import to_ndarray

_TOL = 1e-10


def _is_symmetric(x, tol=_TOL):
    new_x = to_ndarray(x, to_ndim=3)
    return (np.abs(new_x - np.transpose(new_x, axes=(0, 2, 1))) < tol).all()


@custom_vjp
def expm(x):
    x_new = to_ndarray(x, to_ndim=3)
    result = vmap(sp_expm)(x_new)
    return result[0] if len(result) == 1 else result


def _expm_fwd(x):
    return expm(x), x


def _expm_bwd(res, g):
    x = res
    vectorized = x.ndim == 3
    axes = (0, 2, 1) if vectorized else (1, 0)

    n = x.shape[-1]
    size_m = x.shape[:-2] + (2 * n, 2 * n)
    mat = np.zeros(size_m)
    mat[..., :n, :n] = x.transpose(axes)
    mat[..., n:, n:] = x.transpose(axes)
    mat[..., :n, n:] = g
    return expm(mat)[..., :n, n:]


expm.defvjp(expm, _expm_bwd)

logm_prim = core.Primitive('logm')
logm_prim.def_impl(vmap(scipy.linalg.logm))


def logm(x):
    return logm_prim.bind(x)


def solve_sylvester(a, b, q):
    if a.shape == b.shape:
        axes = (0, 2, 1) if a.ndim == 3 else (1, 0)
        if np.all(a == b) and np.all(
                np.abs(a - np.transpose(a, axes)) < 1e-6):
            eigvals, eigvecs = eigh(a)
            if np.all(eigvals >= 1e-6):
                tilde_q = np.transpose(eigvecs, axes) @ q @ eigvecs
                tilde_x = tilde_q / (
                    eigvals[..., :, None] + eigvals[..., None, :])
                return eigvecs @ tilde_x @ np.transpose(eigvecs, axes)

    return np.vectorize(
        scipy.linalg.solve_sylvester,
        signature='(m,m),(n,n),(m,n)->(m,n)')(a, b, q)


def sqrtm(x):
    return np.vectorize(
        scipy.linalg.sqrtm, signature='(n,m)->(n,m)')(x)


def qr(*args, **kwargs):
    return np.vectorize(np.linalg.qr,
                        signature='(n,m)->(n,k),(k,m)')(*args, **kwargs)
