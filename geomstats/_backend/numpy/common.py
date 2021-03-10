import autograd.numpy as np


def to_ndarray(x, to_ndim, axis=0):
    x = np.array(x)
    if x.ndim == to_ndim - 1:
        x = np.expand_dims(x, axis=axis)

    if x.ndim != 0:
        if x.ndim < to_ndim:
            raise ValueError(
                'The ndim was not adapted properly.')
    return x
