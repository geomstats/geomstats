"""Numpy based random backend."""

import numpy as np


def rand(*args, **kwargs):
    return np.random.rand(*args, **kwargs)


def randint(*args, **kwargs):
    return np.random.randint(*args, **kwargs)


def seed(*args, **kwargs):
    return np.random.seed(*args, **kwargs)


def normal(*args, **kwargs):
    return np.random.normal(*args, **kwargs)


def uniform(*args, **kwargs):
    return np.random.uniform(*args, **kwargs)
