"""Tensorflow based random backend."""

import tensorflow as _tf
import tensorflow_probability as _tfp
from tensorflow import cast

from .._dtype_utils import _modify_func_default_dtype

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


@_modify_func_default_dtype(copy=False, kw_only=True)
def rand(*args, dtype=None):
    if dtype in [_tf.complex64, _tf.complex128]:
        real = _tf.cast(_tf.random.uniform(shape=args), dtype=dtype)
        imag = 1j * _tf.cast(_tf.random.uniform(shape=args), dtype=dtype)

        return real + imag
    return _tf.random.uniform(shape=args, dtype=dtype)


def seed(*args):
    return _tf.compat.v1.set_random_seed(*args)


@_modify_func_default_dtype(copy=False, kw_only=False)
def normal(loc=0.0, scale=1.0, size=(1,), dtype=None):
    if not hasattr(size, "__iter__"):
        size = (size,)

    return _tf.random.normal(mean=loc, stddev=scale, shape=size, dtype=dtype)


@_modify_func_default_dtype(copy=False, kw_only=False)
def uniform(low=0.0, high=1.0, size=(1,), dtype=None):
    if not hasattr(size, "__iter__"):
        size = (size,)

    return _tf.random.uniform(shape=size, minval=low, maxval=high, dtype=dtype)


@_modify_func_default_dtype(copy=False, kw_only=False)
def multivariate_normal(mean, cov, size=None, dtype=None):
    if size is None:
        size = ()

    if mean.dtype != dtype:
        mean = cast(mean, dtype)

    if cov.dtype != dtype:
        cov = cast(cov, dtype)

    return _tfd.Sample(
        _tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov),
        sample_shape=size,
    ).sample()
