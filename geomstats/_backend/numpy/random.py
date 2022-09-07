"""Numpy based random backend."""

import numpy as _np
from numpy.random import default_rng as _default_rng
from numpy.random import randint, seed

from ._dtype_wrapper import _add_default_dtype

rand = _add_default_dtype(target=_np.random.rand)
normal = _add_default_dtype(target=_np.random.normal)
multivariate_normal = _add_default_dtype(target=_np.random.multivariate_normal)
uniform = _add_default_dtype(target=_np.random.uniform)


def choice(*args, **kwargs):
    return _default_rng().choice(*args, **kwargs)
