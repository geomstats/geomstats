"""Autograd based random backend."""

import autograd.numpy as _np
from autograd.numpy.random import randint, seed

from ._dtype import _allow_complex_dtype

rand = _allow_complex_dtype(target=_np.random.rand)
uniform = _allow_complex_dtype(target=_np.random.uniform)
normal = _allow_complex_dtype(target=_np.random.normal)
multivariate_normal = _allow_complex_dtype(target=_np.random.multivariate_normal)


def choice(*args, **kwargs):
    return _np.random.default_rng().choice(*args, **kwargs)
