"""Autograd based random backend."""

from autograd.numpy.random import default_rng as _default_rng  # NOQA
from autograd.numpy.random import (
    multivariate_normal,
    normal,
    rand,
    randint,
    seed,
    uniform,
)


def choice(*args, **kwargs):
    return _default_rng().choice(*args, **kwargs)
