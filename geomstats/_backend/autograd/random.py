"""Autograd based random backend."""

from autograd.numpy.random import (  # NOQA
    default_rng,
    multivariate_normal,
    normal,
    rand,
    randint,
    seed,
    uniform,
)


def choice(*args, **kwargs):
    return default_rng().choice(*args, **kwargs)
