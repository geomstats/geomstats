"""Numpy based random backend."""

import numpy as np


def rand(*args, **kwargs):
    return np.random.rand(*args, **kwargs)
