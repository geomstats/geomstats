"""Tensorflow based linear algebra backend."""

import tensorflow as tf


def norm(x):
    return tf.linalg.norm(x)


def inv(x):
    return tf.linalg.inv(x)


def matrix_rank(x):
    return tf.rank(x)


def eigvalsh(x):
    return tf.linalg.eigvalsh(x)
