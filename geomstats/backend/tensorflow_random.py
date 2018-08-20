"""Tensorflow based random backend."""

import tensorflow as tf


def rand(*args):
    return tf.random_uniform(shape=args)


def seed(*args):
    return tf.set_random_seed(*args)


def normal(mean=0.0, std=1.0, shape=(1, 1)):
    tf.random_normal(mean=mean, stddev=std, shape=shape)
