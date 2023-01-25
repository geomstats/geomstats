"""Torch based random backend."""

import torch as _torch
from torch import rand, randint
from torch.distributions.multivariate_normal import (
    MultivariateNormal as _MultivariateNormal,
)

from ._dtype import _allow_complex_dtype, _modify_func_default_dtype


def choice(x, a):
    """Generate a random sample from an array of given size."""
    if _torch.is_tensor(x):
        return x[_torch.randint(len(x), (a,))]

    return x


def seed(*args, **kwargs):
    return _torch.manual_seed(*args, **kwargs)


@_allow_complex_dtype
def normal(loc=0.0, scale=1.0, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _torch.normal(mean=loc, std=scale, size=size)


def uniform(low=0.0, high=1.0, size=(1,), dtype=None):
    if not hasattr(size, "__iter__"):
        size = (size,)
    if low >= high:
        raise ValueError("Upper bound must be higher than lower bound")
    return (high - low) * rand(*size, dtype=dtype) + low


@_modify_func_default_dtype(copy=False, kw_only=True)
@_allow_complex_dtype
def multivariate_normal(mean, cov, size=(1,)):
    if not hasattr(size, "__iter__"):
        size = (size,)
    return _MultivariateNormal(mean, cov).sample(size)
