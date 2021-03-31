"""Tensorflow based random backend."""

import tensorflow as tf


shuffle = tf.random.shuffle


def multivariate_normal(*args, **kwargs):
    return tf.contrib.distributions.MultivariateNormalFullCovariance(
        *args, **kwargs)


def normal(loc=0.0, scale=1.0, size=(1, 1)):
    return tf.random.normal(mean=loc, stddev=scale, shape=size)


def permutation(n):
    return tf.random.shuffle(tf.range(n))


def rand(*args):
    return tf.random.uniform(shape=args)


def choice(x, size, axis=0):
    dim_x = tf.cast(tf.shape(x)[axis], tf.int64)
    indices = tf.range(0, dim_x, dtype=tf.int64)
    sample_index = tf.random.shuffle(indices)[:size]
    sample = tf.gather(x, sample_index, axis=axis)

    return sample


def randint(low, high=None, size=None):
    if size is None:
        size = (1,)
    maxval = high
    minval = low
    if high is None:
        maxval = low - 1
        minval = 0
    return tf.random.uniform(
        shape=size,
        minval=minval,
        maxval=maxval, dtype=tf.int32, seed=None, name=None)


def seed(*args):
    return tf.compat.v1.set_random_seed(*args)


def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        size = (1,)
    if not isinstance(size, tuple):
        size = (size,)
    eager = tf.random.uniform(shape=size, minval=low, maxval=high)
    return tf.Variable(eager)
