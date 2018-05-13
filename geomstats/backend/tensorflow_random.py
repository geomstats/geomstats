"""Tensorflow based random backend."""

import tensorflow as tf


def rand(*args):
    return tf.random_uniform(args)


def seed(*args):
    return tf.set_random_seed(*args)
