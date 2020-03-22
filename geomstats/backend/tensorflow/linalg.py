"""Tensorflow based linear algebra backend."""

import tensorflow as tf

from .common import to_ndarray


def sqrtm(sym_mat):
    sym_mat = to_ndarray(sym_mat, to_ndim=3)

    [eigenvalues, vectors] = tf.linalg.eigh(sym_mat)

    sqrt_eigenvalues = tf.sqrt(eigenvalues)

    aux = tf.einsum('ijk,ik->ijk', vectors, sqrt_eigenvalues)
    sqrt_mat = tf.einsum('ijk,ilk->ijl', aux, vectors)

    sqrt_mat = to_ndarray(sqrt_mat, to_ndim=3)
    return sqrt_mat


def expm(x):
    return tf.linalg.expm(x)


def logm(x):
    x = tf.cast(x, tf.complex64)
    logm = tf.linalg.logm(x)
    logm = tf.cast(logm, tf.float32)
    return logm


def det(x):
    return tf.linalg.det(x)


def eigh(x):
    return tf.linalg.eigh(x)


def eig(x):
    return tf.linalg.eig(x)


def svd(x):
    s, u, v_t = tf.linalg.svd(x, full_matrices=True)
    return u, s, tf.transpose(v_t, perm=(0, 2, 1))


def norm(x, axis=None):
    return tf.linalg.norm(x, axis=axis)


def inv(x):
    return tf.linalg.inv(x)


def matrix_rank(x):
    return tf.rank(x)


def eigvalsh(x):
    return tf.linalg.eigvalsh(x)


def qr(*args, mode='reduced'):
    def qr_aux(x, mode):
        if mode == 'complete':
            aux = tf.linalg.qr(x, full_matrices=True)
        else:
            aux = tf.linalg.qr(x)

        return (aux.q, aux.r)

    qr = tf.map_fn(
        lambda x: qr_aux(x, mode),
        *args,
        dtype=(tf.float32, tf.float32))

    return qr


def powerm(x, power):
    return expm(power * logm(x))
