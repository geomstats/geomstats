"""Numpy based random backend."""

from numpy.random import (  # NOQA
    default_rng,
    normal,
    multivariate_normal,
    rand,
    randint,
    seed,
    uniform
)


def choice(*args, **kwargs):
    return default_rng().choice(*args, **kwargs)
