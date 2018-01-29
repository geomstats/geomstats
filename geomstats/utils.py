""" Utils for the module geomstats."""

import numpy as np


EPSILON = 1e-5


def is_close(x, value, epsilon=EPSILON):
    return np.abs(x - value) < epsilon
