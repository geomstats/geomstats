"""Testing backend."""

import numpy as np


def assert_allclose(*args, **kwargs):
    return np.testing.assert_allclose(*args, **kwargs)
