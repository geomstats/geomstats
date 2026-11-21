"""Autograd based random backend."""

import autograd.numpy as _np
from autograd.numpy.random import randint, seed

from ._dtype import _allow_complex_dtype

rand = _allow_complex_dtype(target=_np.random.rand)
uniform = _allow_complex_dtype(target=_np.random.uniform)
normal = _allow_complex_dtype(target=_np.random.normal)
multivariate_normal = _allow_complex_dtype(target=_np.random.multivariate_normal)


def choice(a, size=None, replace=True, p=None):
    """Generate a random sample from an array, along its first axis.

    It uses the global random state, so that it follows ``seed``.

    Parameters
    ----------
    a : array-like or int
        Array to sample from, along its first axis. If an int, ``arange(a)``.
    size : int or tuple[int]
        Shape of the sample. If None, a single element is returned.
    replace : bool
        Whether the sample is with replacement.
    p : array-like, shape=[len(a)]
        Probability of each element. If None, uniform.

    Returns
    -------
    sample : array-like, shape=[*size, *a.shape[1:]]
        Random sample.
    """
    if isinstance(a, int):
        return _np.random.choice(a, size=size, replace=replace, p=p)

    index = _np.random.choice(len(a), size=size, replace=replace, p=p)
    return a[index]
