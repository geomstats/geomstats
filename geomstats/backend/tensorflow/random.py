"""Tensorflow based random backend."""

import tensorflow as tf


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


def rand(*args):
    return tf.random.uniform(shape=args)


def seed(*args):
    return tf.compat.v1.set_random_seed(*args)


def normal(loc=0.0, scale=1.0, size=(1, 1)):
    return tf.random.normal(mean=loc, stddev=scale, shape=size)
