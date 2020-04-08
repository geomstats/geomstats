"""Numpy based linear algebra backend."""

import autograd.numpy as np
import scipy.linalg
from autograd.numpy.linalg import (  # NOQA
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    norm,
    svd
)

from .common import to_ndarray

_TOL = 1e-10


def _is_symmetric(x, tol=_TOL):
    new_x = to_ndarray(x, to_ndim=3)
    return (np.abs(new_x - np.transpose(new_x, axes=(0, 2, 1))) < tol).all()


def _expsym(x):
    eigvals, eigvecs = np.linalg.eigh(x)
    eigvals = np.exp(eigvals)
    eigvals = np.vectorize(np.diag, signature='(n)->(n,n)')(eigvals)
    transp_eigvecs = np.transpose(eigvecs, axes=(0, 2, 1))
    result = np.matmul(eigvecs, eigvals)
    result = np.matmul(result, transp_eigvecs)
    return result


def expm(x):
    ndim = x.ndim
    new_x = to_ndarray(x, to_ndim=3)
    if _is_symmetric(new_x):
        result = _expsym(new_x)
    else:
        result = np.vectorize(scipy.linalg.expm,
                              signature='(n,m)->(n,m)')(new_x)

    if ndim == 2:
        return result[0]
    return result


def logm(x):
    ndim = x.ndim
    new_x = to_ndarray(x, to_ndim=3)
    if _is_symmetric(new_x):
        eigvals, eigvecs = np.linalg.eigh(new_x)
        if (eigvals > 0).all():
            eigvals = np.log(eigvals)
            eigvals = np.vectorize(np.diag, signature='(n)->(n,n)')(eigvals)
            transp_eigvecs = np.transpose(eigvecs, axes=(0, 2, 1))
            result = np.matmul(eigvecs, eigvals)
            result = np.matmul(result, transp_eigvecs)
        else:
            result = np.vectorize(scipy.linalg.logm,
                                  signature='(n,m)->(n,m)')(new_x)
    else:
        result = np.vectorize(scipy.linalg.logm,
                              signature='(n,m)->(n,m)')(new_x)

    if ndim == 2:
        return result[0]
    return result


def powerm(x, power):
    ndim = x.ndim
    new_x = to_ndarray(x, to_ndim=3)
    if _is_symmetric(new_x):
        eigvals, eigvecs = np.linalg.eigh(new_x)
        if (eigvals > 0).all():
            eigvals = eigvals ** power
            eigvals = np.vectorize(np.diag, signature='(n)->(n,n)')(eigvals)
            transp_eigvecs = np.transpose(eigvecs, axes=(0, 2, 1))
            result = np.matmul(eigvecs, eigvals)
            result = np.matmul(result, transp_eigvecs)
        else:
            log_x = np.vectorize(scipy.linalg.logm,
                                 signature='(n,m)->(n,m)')(new_x)
            p_log_x = power * log_x
            result = np.vectorize(scipy.linalg.expm,
                                  signature='(n,m)->(n,m)')(p_log_x)
    else:
        log_x = np.vectorize(scipy.linalg.logm,
                             signature='(n,m)->(n,m)')(new_x)
        p_log_x = power * log_x
        result = np.vectorize(scipy.linalg.expm,
                              signature='(n,m)->(n,m)')(p_log_x)

    if ndim == 2:
        return result[0]
    return result


def sqrtm(x):
    return np.vectorize(
        scipy.linalg.sqrtm, signature='(n,m)->(n,m)')(x)


def qr(*args, **kwargs):
    return np.vectorize(np.linalg.qr,
                        signature='(n,m)->(n,k),(k,m)',
                        excluded=['mode'])(*args, **kwargs)
