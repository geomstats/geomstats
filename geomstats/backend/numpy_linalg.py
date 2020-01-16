"""Numpy based linear algebra backend."""

import numpy as np
import scipy.linalg

from geomstats.backend.numpy import to_ndarray

def exph(x):
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
    if (new_x - np.transpose(new_x, axes=(0, 2, 1)) == 0).all():
        result = exph(new_x)
    else:
        result = np.vectorize(scipy.linalg.expm,
                               signature='(n,m)->(n,m)')(new_x)

    if ndim == 2:
        return result[0]
    else:
        return result


def logm(x):
    ndim = x.ndim
    new_x = to_ndarray(x, to_ndim=3)
    if (new_x - np.transpose(new_x, axes=(0, 2, 1)) == 0).all():
        eigvals, eigvecs = np.linalg.eigh(new_x)
        if (eigvals > 0).all():
            eigvals = np.vectorize(np.diag, signature='(n)->(n,n)')(np.log(eigvals))
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
    else:
        return result


def powerm(x, power):
    ndim = x.ndim
    new_x = to_ndarray(x, to_ndim=3)
    if (new_x - np.transpose(new_x, axes=(0, 2, 1)) == 0).all():
        eigvals, eigvecs = np.linalg.eigh(new_x)
        if (eigvals > 0).all():
            eigvals = np.vectorize(np.diag, signature='(n)->(n,n)')(eigvals**power)
            transp_eigvecs = np.transpose(eigvecs, axes=(0, 2, 1))
            result = np.matmul(eigvecs, eigvals)
            result = np.matmul(result, transp_eigvecs)
        else:
            result = np.vectorize(scipy.linalg.logm,
                                  signature='(n,m)->(n,m)')(new_x)
            result = power * result
            result = np.vectorize(scipy.linalg.expm,
                                  signature='(n,m)->(n,m)')(result)
    else:
        result = np.vectorize(scipy.linalg.logm,
                              signature='(n,m)->(n,m)')(new_x)
        result = power * result
        result = np.vectorize(scipy.linalg.expm,
                              signature='(n,m)->(n,m)')(result)

    if ndim == 2:
        return result[0]
    else:
        return result


def sqrtm(x):
    return np.vectorize(
        scipy.linalg.sqrtm, signature='(n,m)->(n,m)')(x)


def det(*args, **kwargs):
    return np.linalg.det(*args, **kwargs)


def norm(*args, **kwargs):
    return np.linalg.norm(*args, **kwargs)


def inv(*args, **kwargs):
    return np.linalg.inv(*args, **kwargs)


def matrix_rank(*args, **kwargs):
    return np.linalg.matrix_rank(*args, **kwargs)


def eigvalsh(*args, **kwargs):
    return np.linalg.eigvalsh(*args, **kwargs)


def svd(*args, **kwargs):
    return np.linalg.svd(*args, **kwargs)


def eigh(*args, **kwargs):
    return np.linalg.eigh(*args, **kwargs)


def eig(*args, **kwargs):
    return np.linalg.eig(*args, **kwargs)


def exp(*args, **kwargs):
    return np.exp(*args, **kwargs)


def qr(*args, **kwargs):
    return np.vectorize(
        np.linalg.qr,
        signature='(n,m)->(n,k),(k,m)',
        excluded=['mode'])(*args, **kwargs)
