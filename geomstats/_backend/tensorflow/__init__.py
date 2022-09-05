"""Tensorflow based computation backend."""

from collections import Counter as _Counter
from itertools import product as _product

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp
from tensorflow import abs
from tensorflow import acos as arccos  # NOQA
from tensorflow import acosh as arccosh
from tensorflow import argmax, argmin
from tensorflow import asin as arcsin
from tensorflow import atan2 as arctan2
from tensorflow import broadcast_to, cast
from tensorflow import clip_by_value as clip
from tensorflow import (
    cosh,
    equal,
    exp,
    expand_dims,
    float32,
    float64,
    floor,
    greater,
    int32,
    int64,
    less,
    less_equal,
    logical_and,
    logical_or,
    maximum,
    meshgrid,
    minimum,
    ones_like,
    pad,
)
from tensorflow import reduce_max as amax
from tensorflow import reduce_min as amin
from tensorflow import reduce_prod as prod
from tensorflow import (
    reshape,
    searchsorted,
    sign,
    sinh,
    sort,
    squeeze,
    stack,
    tan,
    tanh,
    uint8,
    zeros_like,
)
from tensorflow.experimental.numpy import empty_like, moveaxis

from .._backend_config import tf_atol as atol
from .._backend_config import tf_rtol as rtol
from . import autodiff  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA
from ._dtype_wrapper import (
    _cast_fout_from_dtype,
    _input_to_tensor_if_float,
    _update_dtype,
    _update_func_default_dtype,
    as_dtype,
    get_default_dtype,
    set_default_dtype,
)

_DTYPES = {
    int32: 0,
    int64: 1,
    float32: 2,
    float64: 3,
    _tf.complex64: 4,
    _tf.complex128: 5,
}

conj = _tf.math.conj
erf = _tf.math.erf
imag = _tf.math.imag
isnan = _tf.math.is_nan
mod = _tf.math.mod
polygamma = _tf.math.polygamma
power = _tf.math.pow
real = _tf.math.real
set_diag = _tf.linalg.set_diag
trapz = _tfp.math.trapz

ones = _update_dtype(_func=_tf.ones, dtype_pos=1)
zeros = _update_dtype(_func=_tf.zeros, dtype_pos=1)
empty = _update_func_default_dtype(_func=_tf.experimental.numpy.empty)

sqrt = _input_to_tensor_if_float(_func=_tf.sqrt)
cos = _input_to_tensor_if_float(_func=_tf.cos)
sin = _input_to_tensor_if_float(_func=_tf.sin)
angle = _input_to_tensor_if_float(_func=_tf.math.angle)
arctanh = _input_to_tensor_if_float(_func=_tf.math.atanh)
ceil = _input_to_tensor_if_float(_func=_tf.math.ceil)
log = _input_to_tensor_if_float(_func=_tf.math.log)


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


def to_numpy(x):
    return x.numpy()


def from_numpy(x):
    return _tf.convert_to_tensor(x)


def arange(start_or_stop, /, stop=None, step=1, dtype=None, **kwargs):

    if dtype is None and (
        type(stop) is float or type(step) is float or type(start_or_stop) is float
    ):
        dtype = get_default_dtype()

    if stop is None:
        return _tf.range(start_or_stop, delta=step, dtype=dtype)

    return _tf.range(start_or_stop, stop, delta=step, dtype=dtype)


def quantile(x, q, axis=None, out=None):
    # Note: numpy, pytorch, and autograd convention is q in [0,1] while tfp expects
    # [0,100]. These other libraries also default to (the equivalent of)
    # interpolation=linear
    result = _tfp.stats.percentile(x, q * 100, axis=axis, interpolation="linear")
    if out is not None:
        out[:] = result
    return result


def one_hot(labels, num_classes):
    return _tf.one_hot(labels, num_classes, dtype=_tf.uint8)


def concatenate(x, axis=0, out=None):
    return _tf.concat(x, axis=axis)


def convert_to_wider_dtype(tensor_list):
    dtype_list = [_DTYPES.get(x.dtype, -1) for x in tensor_list]
    if len(set(dtype_list)) == 1:
        return tensor_list

    wider_dtype_index = max(dtype_list)

    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return tensor_list


def repeat(a, repeats, axis=None):
    return _tf.repeat(input=a, repeats=repeats, axis=axis)


@_cast_fout_from_dtype(dtype_pos=1)
def array(x, dtype=None):
    return _tf.convert_to_tensor(x, dtype=dtype)


def trace(x):
    return _tf.linalg.trace(x)


# TODO (nkoep): Handle the optional axis arguments.
def diagonal(a, axis1=0, axis2=1):
    return _tf.linalg.diag_part(a)


def ndim(x):
    return _tf.convert_to_tensor(x).ndim


def to_ndarray(x, to_ndim, axis=0):
    if ndim(x) == to_ndim - 1:
        x = _tf.expand_dims(x, axis=axis)
    return x


def flip(m, axis=None):
    if not isinstance(m, _tf.Tensor):
        raise ValueError("m must be a Tensorflow tensor")
    if axis is None:
        axis = range(m.ndim)
    elif not hasattr(axis, "__iter__"):
        axis = (axis,)
    return _tf.reverse(m, axis=axis)


def any(x, axis=None):
    return _tf.math.reduce_any(_tf.cast(x, bool), axis=axis)


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return isinstance(x[0], bool)
    if _tf.is_tensor(x):
        return x.dtype == bool
    return False


def _mask_from_indices(indices, mask_shape, dtype=float32):
    """Create a binary mask.

    Parameters
    ----------
    indices: tuple
        Single index or tuple of indices where ones will be.
    mask_shape: tuple
        Shape of the mask.
    dtype: dtype
        Type of the mask.

    Returns
    -------
    _tf_mask : array, shape=[mask_shape]
    """
    np_mask = _np.zeros(mask_shape)

    for i_index, index in enumerate(indices):
        if not isinstance(index, tuple):
            if hasattr(index, "__iter__"):
                indices[i_index] = tuple(index)
            else:
                indices[i_index] = (index,)
    for index in indices:
        if len(index) != len(mask_shape):
            raise ValueError("Indices must have the same size as shape")

    for index in indices:
        np_mask[index] = 1
    _tf_mask = array(np_mask, dtype=dtype)
    return _tf_mask


def _duplicate_array(x, n_samples, axis=0):
    """Stack copies of an array along an additional dimension.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array which will be copied.
    n_samples: int
        Number of copies of the array to create.
    axis: int, optional
        Dimension of the new array along which the copies of x are made.

    Returns
    -------
    tiled_array: array, shape=[dim[:axis], n_samples, dim[axis:]]
        Copies of x stacked along dim axis
    """
    multiples = _np.ones(ndim(x) + 1, dtype=_np.int32)
    multiples[axis] = n_samples
    return tile(to_ndarray(x, ndim(x) + 1, axis), multiples)


def _vectorized_mask_from_indices(
    n_samples=1, indices=None, mask_shape=None, axis=0, dtype=float32
):
    """Create a vectorized binary mask.

    Parameters
    ----------
    n_samples: int
        Number of copies of the mask along the additional dimension.
    indices : {tuple, list(tuple)}
        Single tuple, or list of tuples of indices where ones will be.
    mask_shape : tuple
        Shape of the mask.
    axis: int
        Axis along which the mask is vectorized.
    dtype: dtype
        Type of the returned array.

    Returns
    -------
    _tf_mask : array, shape=[mask_shape[:axis], n_samples, mask_shape[axis:]]
    """
    mask = _mask_from_indices(indices, mask_shape, dtype)
    return _duplicate_array(mask, n_samples, axis=axis)


def _assignment_single_value(x, value, indices, mode="replace", axis=0):
    """Assign a value at given indices of an array.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    value : float
        Value to be added.
    indices : {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), value is
        assigned to each copy along axis.
    mode : string, optional
        Whether the assignment is done by replacing the old value,
        or by adding to it. Possible values are 'replace' and 'sum'
    axis : int, optional
        Axis along which value is assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x where value was assigned at all indices (and possibly
        along an axis).
    """
    single_index = not isinstance(indices, list)
    if _tf.is_tensor(indices):
        single_index = ndim(indices) <= 1 and sum(indices.shape) <= ndim(x)
    if single_index:
        indices = [indices]

    if isinstance(indices[0], tuple):
        use_vectorization = len(indices[0]) < ndim(x)
    elif _tf.is_tensor(indices[0]) and ndim(indices[0]) >= 1:
        use_vectorization = len(indices[0]) < ndim(x)
    else:
        use_vectorization = ndim(x) > 1

    if use_vectorization:
        full_shape = shape(x)
        n_samples = full_shape[axis]
        tile_shape = list(full_shape[:axis]) + list(full_shape[axis + 1 :])
        mask = _vectorized_mask_from_indices(
            n_samples, indices, tile_shape, axis, x.dtype
        )
    else:
        mask = _mask_from_indices(indices, shape(x), x.dtype)
    if mode == "replace":
        return x + -x * mask + value * mask
    if mode == "sum":
        return x + value * mask
    raise ValueError("mode must be one of 'replace' or 'sum'")


def _assignment(x, values, indices, mode, axis):
    if _is_boolean(indices):
        if ndim(array(indices)) > 1:
            indices_tensor = _tf.where(indices)
            indices = [tuple(ind) for ind in indices_tensor]
        else:
            indices_from_booleans = [index for index, val in enumerate(indices) if val]
            indices_along_dims = [range(dim) for dim in shape(x)]
            indices_along_dims[axis] = indices_from_booleans
            indices = list(_product(*indices_along_dims))
    if _tf.rank(values) == 0:
        return _assignment_single_value(x, values, indices, mode, axis)
    values = cast(flatten(array(values)), x.dtype)

    single_index = not isinstance(indices, list)
    if _tf.is_tensor(indices):
        single_index = ndim(indices) <= 1 and sum(indices.shape) <= ndim(x)
    if single_index:
        if len(values) > 1:
            indices = [
                tuple(list(indices[:axis]) + [i] + list(indices[axis:]))
                for i in range(x.shape[axis])
            ]
        else:
            indices = [indices]

    if len(values) != len(indices):
        raise ValueError("Either one value or as many values as indices")

    for i_index, index in enumerate(indices):
        x = _assignment_single_value(x, values[i_index], index, mode, axis)
    return x


def assignment(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {
    int, tuple(int), array-like({int, tuple, boolean})
        Single index or array of indices where values are assigned.
        If the length of the tuples is shorter than ndim(x) by one, values are
        assigned to each copy along axis.
        If indices is a list of booleans and ndim(x) > 1, values are assigned
        across all dimensions.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x as the sum of x and the values at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a single index is provided, and len(indices) == ndim(x) - 1, then values
    are assigned along axis.

    Examples
    --------
    Most examples translate as
    assignment(x, indices, values) <=> x[indices] = values
    Some special cases are given by vectorisation.
    (Beware that copies are always returned).
    if ndim(x) == 3, assignment(x, 1, (1, 0), 1) <=> x[1, :, 0] = 1
    if ndim(x) == 2, assignment(x, [1, 2], [(0, 1), (2, 3)]) <=>
                        x[((0, 2), (1, 3))] = [1, 2]
    """
    return _assignment(x, values, indices, "replace", axis)


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {
    int, tuple(int), array-like({int, tuple, boolean})
        Single index or array of indices where values are assigned.
        If the length of the tuples is shorter than ndim(x) by one, values are
        assigned to each copy along axis.
        If indices is a list of booleans and ndim(x) > 1, values are assigned
        across all dimensions.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x as the sum of x and the values at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a single index is provided, and len(indices) == ndim(x) - 1, then values
    are assigned along axis.

    Examples
    --------
    Most examples translate as
    assignment_by_sum(x, indices, values) <=> x[indices] = x[indices] + values
    Some special cases are given by vectorisation.
    (Beware that copies are always returned).
    if ndim(x) == 3, assignment_by_sum(x, 1, (1, 0), 1) <=> x[1, :, 0] += 1
    if ndim(x) == 2, assignment_by_sum(x, [1, 2], [(0, 1), (2, 3)]) <=>
                        x[((0, 2), (1, 3))] += [1, 2]
    """
    return _assignment(x, values, indices, "sum", axis)


def array_from_sparse(indices, data, target_shape):
    """Create an array of given shape, with values at specific indices.

    The rest of the array will be filled with zeros.

    Parameters
    ----------
    indices : iterable(tuple(int))
        Index of each element which will be assigned a specific value.
    data : iterable(scalar)
        Value associated at each index.
    target_shape : tuple(int)
        Shape of the output array.

    Returns
    -------
    a : array, shape=target_shape
        Array of zeros with specified values assigned to specified indices.
    """
    data = array(data)
    return _tf.sparse.to_dense(
        _tf.sparse.reorder(_tf.SparseTensor(indices, data, target_shape))
    )


def get_slice(x, indices):
    """Return a slice of an array, following Numpy's style.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    indices : iterable(iterable(int))
        Indices which are kept along each axis, starting from 0.

    Returns
    -------
    slice : array-like
        Slice of x given by indices.

    Notes
    -----
    This follows Numpy's convention: indices are grouped by axis.

    Examples
    --------
    >>> a = tf.reshape(_tf.convert_to_tensor(range(30)), (3,10))
    >>> get_slice(a, ((0, 2), (8, 9)))
    <tf.Tensor: id=41, shape=(2,), dtype=int32, numpy=array([ 8, 29])>
    """
    if hasattr(indices, "shape"):
        if indices.shape.rank == 0:
            return x[indices]

        if _tf.is_tensor(indices) and indices.shape[-1] == 1:
            return _tf.gather_nd(x, indices)

    return _tf.gather_nd(x, list(zip(*indices)))


def vectorize(x, pyfunc, multiple_args=False, dtype=None, **kwargs):
    if multiple_args:
        return _tf.map_fn(lambda y: pyfunc(*y), elems=x, dtype=dtype)
    return _tf.map_fn(pyfunc, elems=x, dtype=dtype)


def split(x, indices_or_sections, axis=0):
    if isinstance(indices_or_sections, int):
        return _tf.split(x, indices_or_sections, axis=axis)
    indices_or_sections = _np.array(indices_or_sections)
    intervals_length = indices_or_sections[1:] - indices_or_sections[:-1]
    last_interval_length = x.shape[axis] - indices_or_sections[-1]
    if last_interval_length > 0:
        intervals_length = _np.append(intervals_length, last_interval_length)
    intervals_length = _np.insert(intervals_length, 0, indices_or_sections[0])
    return _tf.split(x, num_or_size_splits=tuple(intervals_length), axis=axis)


def hsplit(x, n_splits):
    return _tf.split(x, num_or_size_splits=n_splits, axis=1)


def flatten(x):
    """Collapse the tensor into 1-D.

    Following https://www.tensorflow.org/api_docs/python/_tf/reshape
    """
    return _tf.reshape(x, [-1])


def outer(x, y):
    return einsum("...i,...j->...ij", x, y)


def copy(x):
    return _tf.Variable(x)


def hstack(x):
    return _tf.concat(x, axis=1)


def vstack(x):
    new_x = []
    for one_x in x:
        if one_x.ndim < 2:
            new_x.append(_tf.expand_dims(one_x, axis=0))
        else:
            new_x.append(one_x)
    return _tf.concat(new_x, axis=0)


def broadcast_arrays(*args, **kwargs):
    tensors = [*args]
    shapes = [t.get_shape().as_list() for t in tensors]
    max_rank = max(len(s) for s in shapes)

    for index, value in enumerate(shapes):
        if len(value) == max_rank:
            continue

        tensor = tensors[index]
        for _ in range(max_rank - len(value)):
            value.insert(0, 1)
            tensor = _tf.expand_dims(tensor, axis=0)
        tensors[index] = tensor

    broadcast_shape = []
    for index in range(max_rank):
        dimensions = [s[index] for s in shapes]
        repeats = _Counter(dimensions)
        if len(repeats) > 2 or (len(repeats) == 2 and 1 not in list(repeats.keys())):
            raise ValueError(
                "operands could not be broadcast together with shapes", shapes
            )
        broadcast_shape.append(max(repeats.keys()))

    for axis, dimension in enumerate(broadcast_shape):
        tensors = [
            _tf.concat([t] * dimension, axis=axis) if t.get_shape()[axis] == 1 else t
            for t in tensors
        ]

    return tensors


def dot(a, b):
    if b.ndim == 1:
        return _tf.tensordot(*convert_to_wider_dtype([a, b]), axes=1)

    return einsum("...i,...i->...", a, b)


def isclose(x, y, rtol=rtol, atol=atol):
    if not _tf.is_tensor(x):
        x = _tf.constant(x)
    if not _tf.is_tensor(y):
        y = _tf.constant(y)
    x, y = convert_to_wider_dtype([x, y])
    dtype = x.dtype

    rhs = _tf.constant(atol, dtype=dtype) + _tf.constant(rtol, dtype=dtype) * _tf.abs(y)
    return _tf.less_equal(_tf.abs(_tf.subtract(x, y)), rhs)


def allclose(x, y, rtol=rtol, atol=atol):
    return _tf.reduce_all(isclose(x, y, rtol=rtol, atol=atol))


@_update_func_default_dtype(copy=False)
def eye(n, m=None, dtype=None):
    if m is None:
        m = n
    return _tf.eye(num_rows=n, num_columns=m, dtype=dtype)


def sum(x, axis=None, dtype=None, keepdims=False):
    if not _tf.is_tensor(x):
        x = array(x)

    if dtype is not None and x.dtype != dtype:
        x = cast(x, dtype)

    if x.dtype == bool:
        x = cast(x, int32)

    return _tf.reduce_sum(x, axis=axis, keepdims=keepdims)


def std(x, axis=None, dtype=None, keepdims=False):
    if dtype is not None and x.dtype != dtype:
        x = cast(x, dtype)

    return _tf.math.reduce_std(x, axis=axis, keepdims=keepdims)


def einsum(equation, *inputs):
    input_tensors_list = [arg if is_array(arg) else array(arg) for arg in inputs]
    input_tensors_list = convert_to_wider_dtype(input_tensors_list)

    return _tf.einsum(equation, *input_tensors_list)


def transpose(x, axes=None):
    return _tf.transpose(x, perm=axes)


def all(x, axis=None):
    return _tf.math.reduce_all(_tf.cast(x, bool), axis=axis)


def cumsum(a, axis=None, dtype=None):
    if dtype is not None and a.dtype != dtype:
        a = cast(a, dtype)

    if axis is None:
        return _tf.math.cumsum(flatten(a), axis=0)
    return _tf.math.cumsum(a, axis=axis)


def cumprod(a, axis=None, dtype=None):
    if dtype is not None and a.dtype != dtype:
        a = cast(a, dtype)

    if axis is None:
        return _tf.math.cumprod(flatten(a), axis=0)
    return _tf.math.cumprod(a, axis=axis)


def mean(a, axis=None, dtype=None):
    if dtype is not None and a.dtype != dtype:
        a = cast(a, dtype)

    return _tf.reduce_mean(a, axis=axis)


# (sait) there is _tf.experimental.tril (we can use it once it moves to stable)
def tril(mat, k=0):
    if k not in (0, -1):
        raise NotImplementedError("Only k=0 and k=-1 supported so far")
    tril = _tf.linalg.band_part(mat, -1, 0)
    if k == 0:
        return tril
    zero_diag = _tf.zeros(mat.shape[:-1], dtype=mat.dtype)
    return _tf.linalg.set_diag(tril, zero_diag)


# TODO(sait) use _tf.experimental.triu once it becomes stable.
def triu(mat, k=0):
    if k not in (0, 1):
        raise NotImplementedError("Only k=0 and k=1 supported so far")
    triu = _tf.linalg.band_part(mat, 0, -1)
    if k == 0:
        return triu
    zero_diag = _tf.zeros(mat.shape[:-1], dtype=mat.dtype)
    return _tf.linalg.set_diag(triu, zero_diag)


def diag_indices(*args, **kwargs):
    return tuple(map(_tf.convert_to_tensor, _np.diag_indices(*args, **kwargs)))


def tril_indices(*args, **kwargs):
    return tuple(map(_tf.convert_to_tensor, _np.tril_indices(*args, **kwargs)))


def triu_indices(*args, **kwargs):
    return tuple(map(_tf.convert_to_tensor, _np.triu_indices(*args, **kwargs)))


def unique(x):
    return _tf.unique(x).y


def where(condition, x=None, y=None):
    if x is None and y is None:
        return _tf.where(condition)

    if type(x) is float:
        x = _tf.constant(x, dtype=get_default_dtype())

    out = _tf.where(condition, x, y)

    return out


def tril_to_vec(x, k=0):
    n = x.shape[-1]
    axis = 1 if x.ndim == 3 else 0
    mask = _tf.ones((n, n))
    mask_a = _tf.linalg.band_part(mask, -1, 0)
    if k < 0:
        mask_b = _tf.linalg.band_part(mask, -k - 1, 0)
    else:
        mask_b = _tf.zeros_like(mask_a)
    mask = _tf.cast(mask_a - mask_b, dtype=_tf.bool)
    return _tf.boolean_mask(x, mask, axis=axis)


def triu_to_vec(x, k=0):
    n = x.shape[-1]
    axis = 1 if x.ndim == 3 else 0
    mask = _tf.ones((n, n))
    mask_a = _tf.linalg.band_part(mask, 0, -1)
    if k > 0:
        mask_b = _tf.linalg.band_part(mask, 0, k - 1)
    else:
        mask_b = _tf.zeros_like(mask_a)
    mask = _tf.cast(mask_a - mask_b, dtype=_tf.bool)
    return _tf.boolean_mask(x, mask, axis=axis)


def tile(x, multiples):
    t1 = _tf.ones(len(multiples) - len(_tf.shape(x)))
    t1 = _tf.cast(t1, _tf.int32)
    t2 = _tf.shape(x)
    x_reshape = _tf.reshape(x, _tf.concat([t1, t2], axis=0))
    return _tf.tile(x_reshape, multiples)


def vec_to_diag(vec):
    return _tf.linalg.diag(vec)


def mat_from_diag_triu_tril(diag, tri_upp, tri_low):
    """Build matrix from given components.

    Forms a matrix from diagonal, strictly upper triangular and
    strictly lower traingular parts.

    Parameters
    ----------
    diag : array_like, shape=[..., n]
    tri_upp : array_like, shape=[..., (n * (n - 1)) / 2]
    tri_low : array_like, shape=[..., (n * (n - 1)) / 2]

    Returns
    -------
    mat : array_like, shape=[..., n, n]
    """
    diag, tri_upp, tri_low = convert_to_wider_dtype([diag, tri_upp, tri_low])

    n = diag.shape[-1]
    (i,) = _np.diag_indices(n, ndim=1)
    j, k = _np.triu_indices(n, k=1)

    if diag.ndim == 1:
        upper_indices = [(jj, kk) for jj, kk in zip(j, k)]
        lower_indices = [(kk, jj) for jj, kk in zip(j, k)]
    else:
        m = diag.shape[0]
        upper_indices = [(rr, jj, kk) for rr in range(m) for jj, kk in zip(j, k)]
        lower_indices = [(rr, kk, jj) for rr in range(m) for jj, kk in zip(j, k)]

    mat = zeros(diag.shape + (n,), dtype=diag.dtype)
    mat = assignment(mat, tri_upp, upper_indices)
    mat = assignment(mat, tri_low, lower_indices)

    mat = _tf.linalg.set_diag(mat, diag)

    return mat


def divide(a, b, ignore_div_zero=False):
    a, b = convert_to_wider_dtype([a, b])
    if ignore_div_zero is False:
        return _tf.math.divide(a, b)
    return _tf.math.divide_no_nan(a, b)


def _ravel_multi_index(multi_index, shape):
    strides = _tf.math.cumprod(shape, exclusive=True, reverse=True)
    return _tf.reduce_sum(multi_index * _tf.expand_dims(strides, 1), axis=0)


def ravel_tril_indices(n, k=0, m=None):
    if m is None:
        size = (n, n)
    else:
        size = (n, m)
    idxs = tril_indices(n, k, m)
    return _ravel_multi_index(idxs, size)


def kron(a, b):
    return _tf.linalg.LinearOperatorKronecker([a, b]).to_dense()


def take(a, indices, axis=0):
    return _tf.gather(a, indices, axis=axis)


@_cast_fout_from_dtype(dtype_pos=3)
def linspace(start, stop, num=50, dtype=None):
    return _tf.linspace(start, stop, num)


def is_array(x):
    return _tf.is_tensor(x)


def matvec(A, b):
    A, b = convert_to_wider_dtype([A, b])
    return _tf.linalg.matvec(A, b)


def cross(a, b):
    if a.ndim + b.ndim == 3 or a.ndim == b.ndim == 2 and a.shape[0] != b.shape[0]:
        a, b = broadcast_arrays(a, b)
    return _tf.linalg.cross(*convert_to_wider_dtype([a, b]))


def shape(a):
    if not is_array(a):
        a = array(a)

    return tuple(a.shape)


def matmul(x, y):
    for array_ in [x, y]:
        if array_.ndim == 1:
            raise ValueError("ndims must be >=2")

    x, y = convert_to_wider_dtype([x, y])
    return _tf.linalg.matmul(x, y)
