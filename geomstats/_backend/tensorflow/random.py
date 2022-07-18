"""Tensorflow based random backend."""

import tensorflow as _tf
import tensorflow_probability as _tfp

_tfd = _tfp.distributions


def choice(x, size, axis=0):
    dim_x = _tf.cast(_tf.shape(x)[axis], _tf.int64)
    indices = _tf.range(0, dim_x, dtype=_tf.int64)
    sample_index = _tf.random.shuffle(indices)[:size]
    sample = _tf.gather(x, sample_index, axis=axis)

    return sample


def randint(low, high=None, size=None):
    if size is None:
        size = (1,)
    maxval = high
    minval = low
    if high is None:
        maxval = low - 1
        minval = 0
    return _tf.random.uniform(
        shape=size, minval=minval, maxval=maxval, dtype=_tf.int32, seed=None, name=None
    )


def rand(*args):
    return _tf.random.uniform(shape=args)


def seed(*args):
    return _tf.compat.v1.set_random_seed(*args)


def normal(loc=0.0, scale=1.0, size=(1, 1)):
    return _tf.random.normal(mean=loc, stddev=scale, shape=size)


def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        size = (1,)
    return _tf.random.uniform(shape=size, minval=low, maxval=high)


def multivariate_normal(mean, cov, size=None):
    if size is None:
        size = ()
    return _tfd.Sample(
        _tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov),
        sample_shape=size,
    ).sample()
