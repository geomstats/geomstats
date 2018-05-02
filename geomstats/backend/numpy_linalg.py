"""Numpy based linear algebra backend."""

import numpy as np


def norm(*args, **kwargs):
    return np.linalg.norm(*args, **kwargs)
