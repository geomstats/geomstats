"""Numpy based linear algebra backend."""

import numpy as np


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
