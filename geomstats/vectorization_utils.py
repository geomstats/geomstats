"""
Utils to factorize geomstats code w.r.t. vectorization.
"""

import numpy as np


def expand_dims(element, to_dim, axis=0):
    if element.ndim == to_dim - 1:
        element = np.expand_dims(element, axis=axis)
    assert element.ndim == to_dim
    return element
