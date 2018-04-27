"""
Utils to factorize geomstats code w.r.t. vectorization.
"""

import keras.backend as K

def to_ndarray(element, to_ndim, axis=0):
    if K.ndim(element) == to_ndim - 1:
        element = K.expand_dims(element, axis=axis)
    return element
