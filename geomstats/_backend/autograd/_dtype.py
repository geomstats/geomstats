import autograd.numpy as _np

from geomstats._backend._dtype_utils import _dyn_update_dtype
from geomstats._backend._dtype_utils import _np_box_binary_scalar as _box_binary_scalar
from geomstats._backend._dtype_utils import _np_box_unary_scalar as _box_unary_scalar
from geomstats._backend._dtype_utils import (
    _pre_add_default_dtype_by_casting,
    _pre_cast_fout_to_input_dtype,
    _pre_cast_out_from_dtype,
    _pre_set_default_dtype,
    get_default_cdtype,
    get_default_dtype,
)

from ._common import cast


def is_float(x):
    return x.dtype.kind == "f"


def is_complex(x):
    return x.dtype.kind == "c"


def as_dtype(value):
    """Transform string representing dtype in dtype."""
    return _np.dtype(value)


set_default_dtype = _pre_set_default_dtype(as_dtype)


_add_default_dtype_by_casting = _pre_add_default_dtype_by_casting(cast)
_cast_fout_to_input_dtype = _pre_cast_fout_to_input_dtype(cast, is_float)
_cast_out_from_dtype = _pre_cast_out_from_dtype(cast, is_float, is_complex)
