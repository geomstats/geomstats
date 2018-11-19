"""Tensorflow based linear algebra backend."""

import tensorflow as tf


def det(x):
    return tf.linalg.det(x)


def eigh(x):
    return tf.linalg.eigh(x)


def eig(x):
    return tf.linalg.eig(x)


def svd(x):
    s, u, v_t = tf.svd(x, full_matrices=True)
    return u, s, tf.transpose(v_t, perm=(0, 2, 1))


def norm(x, axis=None):
    return tf.linalg.norm(x, axis=axis)


def inv(x):
    return tf.linalg.inv(x)


def matrix_rank(x):
    return tf.rank(x)


def eigvalsh(x):
    return tf.linalg.eigvalsh(x)


def qr(x):
    return tf.linalg.qr(x)
