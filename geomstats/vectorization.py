"""
Utils to factorize geomstats code w.r.t. vectorization.
"""

import numpy as np


def to_ndarray(element, to_ndim, axis=0):
    element = np.asarray(element)

    if element.ndim == to_ndim - 1:
        element = np.expand_dims(element, axis=axis)
    assert element.ndim == to_ndim
    return element
