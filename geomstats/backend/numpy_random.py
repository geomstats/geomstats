"""Numpy based random backend."""

import numpy as np


def rand(*args, **kwargs):
    return np.random.rand(*args, **kwargs)


def randint(*args, **kwargs):
    return np.random.randint(*args, **kwargs)


def seed(*args, **kwargs):
    return np.random.seed(*args, **kwargs)


def normal(mean=0.0, std=1.0, shape=(1, 1)):
    return np.random.normal(loc=mean, scale=std, size=shape)