"""Torch based random backend."""

import math as _math

import torch as _torch
from torch import rand, randint
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype import _allow_complex_dtype


def choice(a, size=None, replace=True, p=None):
    """Generate a random sample from an array, along its first axis.

    Parameters
    ----------
    a : array-like or int
        Array to sample from, along its first axis. If an int, ``arange(a)``.
    size : int or tuple[int]
        Shape of the sample. If None, a single element is returned.
    replace : bool
        Whether the sample is with replacement.
    p : array-like, shape=[len(a)]
        Probability of each element. If None, uniform.

    Returns
    -------
    sample : array-like, shape=[*size, *a.shape[1:]]
        Random sample.
    """
    if isinstance(a, int):
        a = _torch.arange(a)

    shape = _shape(size)
    n_samples = _math.prod(shape)
    if p is None and replace:
        index = _torch.randint(len(a), (n_samples,), device=a.device)
    else:
        weights = (
            _torch.ones(len(a), device=a.device)
            if p is None
            else _torch.as_tensor(p, device=a.device)
        )
        index = _torch.multinomial(weights, n_samples, replacement=replace)
    return a[_torch.reshape(index, shape)]


def _shape(size):
    """Get the shape of a sample, a scalar one if size is None."""
    if size is None:
        return ()
    return tuple(size) if hasattr(size, "__iter__") else (size,)


def seed(*args, **kwargs):
    return _torch.manual_seed(*args, **kwargs)


@_allow_complex_dtype
def normal(loc=0.0, scale=1.0, size=None):
    return _torch.normal(mean=loc, std=scale, size=_shape(size))


def uniform(low=0.0, high=1.0, size=None, dtype=None):
    if low >= high:
        raise ValueError("Upper bound must be higher than lower bound")
    return (high - low) * rand(_shape(size), dtype=dtype) + low


@_allow_complex_dtype
def multivariate_normal(mean, cov, size=None):
    return _MultivariateNormal(mean, cov).sample(_shape(size))
