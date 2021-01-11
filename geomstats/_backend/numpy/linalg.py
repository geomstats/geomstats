"""Numpy based linear algebra backend."""

import autograd.numpy as np
import autograd.scipy.linalg as asp
from autograd.extend import defvjp, primitive
from autograd.numpy.linalg import (  # NOQA
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    norm,
    svd
)


_TOL = 1e-10


@primitive
def expm(x):
    return np.vectorize(
        asp.expm, signature='(n,m)->(n,m)')(x)


def _expm_vjp(_ans, x):
    vectorized = x.ndim == 3
    axes = (0, 2, 1) if vectorized else (1, 0)

    def vjp(g):
        n = x.shape[-1]
        size_m = x.shape[:-2] + (2 * n, 2 * n)
        mat = np.zeros(size_m)
        mat[..., :n, :n] = x.transpose(axes)
        mat[..., n:, n:] = x.transpose(axes)
        mat[..., :n, n:] = g
        return expm(mat)[..., :n, n:]
    return vjp


defvjp(expm, _expm_vjp)


def logm(x):
    return np.vectorize(
        asp.logm, signature='(n,m)->(n,m)')(x)


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
        asp.solve_sylvester,
        signature='(m,m),(n,n),(m,n)->(m,n)')(a, b, q)


def sqrtm(x):
    return np.vectorize(
        asp.sqrtm, signature='(n,m)->(n,m)')(x)


def qr(*args, **kwargs):
    return np.vectorize(np.linalg.qr,
                        signature='(n,m)->(n,k),(k,m)',
                        excluded=['mode'])(*args, **kwargs)
