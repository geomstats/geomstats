from ._dispatch import _common
from ._dispatch import numpy as _np

_modify_func_default_dtype = _common._modify_func_default_dtype
_allow_complex_dtype = _common._allow_complex_dtype


rand = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.rand)
)

uniform = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.uniform)
)


normal = _modify_func_default_dtype(
    copy=False, kw_only=True, target=_allow_complex_dtype(target=_np.random.normal)
)

multivariate_normal = _modify_func_default_dtype(
    copy=False,
    kw_only=True,
    target=_allow_complex_dtype(target=_np.random.multivariate_normal),
)


def choice(*args, **kwargs):
    return _np.random.default_rng().choice(*args, **kwargs)
