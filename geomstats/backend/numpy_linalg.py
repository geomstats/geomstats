"""Numpy based linear algebra backend."""

import numpy as np
import scipy.linalg


def expm(x):
    return np.vectorize(
        scipy.linalg.expm, signature='(n,m)->(n,m)')(x)


def logm(x):
    return np.vectorize(
        scipy.linalg.logm, signature='(n,m)->(n,m)')(x)


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
