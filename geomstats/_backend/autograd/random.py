"""Autograd based random backend."""
import autograd.numpy as _np
from autograd.numpy.random import default_rng as _default_rng  # NOQA
from autograd.numpy.random import randint, seed

from ._common import cast as _cast
from ._dtype import _add_default_dtype_by_casting, _modify_func_default_dtype

normal = _add_default_dtype_by_casting(target=_np.random.normal)
multivariate_normal = _add_default_dtype_by_casting(
    target=_np.random.multivariate_normal
)
uniform = _add_default_dtype_by_casting(target=_np.random.uniform)


@_modify_func_default_dtype(copy=False, kw_only=True)
def rand(*size, dtype=None):
    if dtype in [_np.complex64, _np.complex128]:
        real = _np.random.rand(*size)
        imag = 1j * _np.random.rand(*size)
        out = real + imag

    else:
        out = _np.random.rand(*size)

    if out.dtype != dtype:
        return _cast(out, dtype)

    return out


def choice(*args, **kwargs):
    return _default_rng().choice(*args, **kwargs)
