"""Tensorflow based random backend."""

import tensorflow as tf


def rand(shape):
    return tf.random_uniform(shape)
