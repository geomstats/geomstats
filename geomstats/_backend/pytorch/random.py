"""Torch based random backend."""

import torch as _torch
from torch import rand, randint
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype_wrapper import _add_default_dtype_by_casting


def choice(x, a):
    """Generate a random sample from an array of given size."""
    if _torch.is_tensor(x):
        return x[_torch.randint(len(x), (a,))]

    return x


def seed(*args, **kwargs):
    return _torch.manual_seed(*args, **kwargs)


@_add_default_dtype_by_casting
def normal(loc=0.0, scale=1.0, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _torch.normal(mean=loc, std=scale, size=size)


@_add_default_dtype_by_casting
def uniform(low=0.0, high=1.0, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    if low >= high:
        raise ValueError("Upper bound must be higher than lower bound")
    return (high - low) * _torch.rand(*size) + low


@_add_default_dtype_by_casting
def multivariate_normal(mean, cov, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _MultivariateNormal(mean, cov).sample(size)
