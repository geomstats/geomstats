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
from tensorflow import broadcast_to
from tensorflow import clip_by_value as clip
from tensorflow import (
    cos,
    cosh,
    divide,
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
    ones,
    ones_like,
    pad,
)
from tensorflow import range as arange
from tensorflow import reduce_max as amax
from tensorflow import reduce_mean as mean
from tensorflow import reduce_min as amin
from tensorflow import reduce_prod as prod
from tensorflow import (
    reshape,
    searchsorted,
    shape,
    sign,
    sin,
    sinh,
    sort,
    sqrt,
    squeeze,
    stack,
    tan,
    tanh,
    uint8,
    zeros,
    zeros_like,
)
from tensorflow.experimental.numpy import moveaxis

from .._backend_config import tf_atol as atol
from .._backend_config import tf_rtol as rtol
from . import autodiff  # NOQA
from . import linalg  # NOQA
from . import random  # NOQA

_DTYPES = {
    int32: 0,
    int64: 1,
    float32: 2,
    float64: 3,
    _tf.complex64: 4,
    _tf.complex128: 5,
}

angle = _tf.math.angle
arctanh = _tf.math.atanh
ceil = _tf.math.ceil
complex64 = _tf.dtypes.complex64
complex128 = _tf.dtypes.complex128
conj = _tf.math.conj
cross = _tf.linalg.cross
erf = _tf.math.erf
imag = _tf.math.imag
isnan = _tf.math.is_nan
log = _tf.math.log
mod = _tf.math.mod
polygamma = _tf.math.polygamma
power = _tf.math.pow
real = _tf.math.real
set_diag = _tf.linalg.set_diag
std = _tf.math.reduce_std
trapz = _tfp.math.trapz


def _raise_not_implemented_error(*args, **kwargs):
    raise NotImplementedError


def to_numpy(x):
    return x.numpy()


def from_numpy(x):
    return _tf.convert_to_tensor(x)


def one_hot(labels, num_classes):
    return _tf.one_hot(labels, num_classes, dtype=_tf.uint8)


def concatenate(x, axis=0, out=None):
    return _tf.concat(x, axis=axis)


def convert_to_wider_dtype(tensor_list):
    dtype_list = [_DTYPES[x.dtype] for x in tensor_list]
    wider_dtype_index = max(dtype_list)

    wider_dtype = list(_DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, dtype=wider_dtype) for x in tensor_list]
    return tensor_list


def repeat(a, repeats, axis=None):
    return _tf.repeat(input=a, repeats=repeats, axis=axis)


def array(x, dtype=None):
    return _tf.convert_to_tensor(x, dtype=dtype)


def trace(x, axis1=0, axis2=1):
    min_axis = min(axis1, axis2)
    max_axis = max(axis1, axis2)
    if min_axis == 1 and max_axis == 2:
        return _tf.einsum("...ii", x)
    if min_axis == -2 and max_axis == -1:
        return _tf.einsum("...ii", x)
    if min_axis == 0 and max_axis == 1:
        return _tf.einsum("ii...", x)
    if min_axis == 0 and max_axis == 2:
        return _tf.einsum("i...i", x)
    raise NotImplementedError()


# TODO (nkoep): Handle the optional axis arguments.
def diagonal(a, axis1=0, axis2=1):
    return _tf.linalg.diag_part(a)


def ndim(x):
    return _tf.convert_to_tensor(x).ndim


def to_ndarray(x, to_ndim, axis=0):
    if ndim(x) == to_ndim - 1:
        x = _tf.expand_dims(x, axis=axis)
    return x


def empty(shape, dtype=float64):
    if not isinstance(dtype, _tf.DType):
        raise ValueError("dtype must be one of Tensorflow's types")
    np_dtype = dtype.as_numpy_dtype
    return _tf.convert_to_tensor(_np.empty(shape, dtype=np_dtype))


def empty_like(prototype, dtype=None):
    initial_shape = _tf.shape(prototype)
    if dtype is None:
        dtype = prototype.dtype
    return empty(initial_shape, dtype=dtype)


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


def get_mask_i_float(i, n):
    """Create a 1D array of zeros with one element at one, with floating type.

    Parameters
    ----------
    i : int
        Index of the non-zero element.
    n: n
        Length of the created array.

    Returns
    -------
    mask_i_float : array-like, shape=[n,]
        1D array of zeros except at index i, where it is one
    """
    range_n = arange(n)
    i_float = cast(array([i]), int32)[0]
    mask_i = equal(range_n, i_float)
    mask_i_float = cast(mask_i, float32)
    return mask_i_float


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
        full_shape = shape(x).numpy()
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
    if type(data) is list:
        data = _tf.convert_to_tensor(_np.array(data))
    data_type = data.dtype
    if data_type is _tf.dtypes.complex64 or data_type is _tf.dtypes.complex128:
        a = _tf.cast(
            _tf.sparse.to_dense(
                _tf.sparse.reorder(
                    _tf.SparseTensor(indices, _tf.math.real(data), target_shape)
                )
            ),
            dtype=data_type,
        ) + 1j * _tf.cast(
            _tf.sparse.to_dense(
                _tf.sparse.reorder(
                    _tf.SparseTensor(indices, _tf.math.imag(data), target_shape)
                )
            ),
            dtype=data_type,
        )
    else:
        a = _tf.sparse.to_dense(
            _tf.sparse.reorder(_tf.SparseTensor(indices, data, target_shape))
        )
    return a


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


def matmul(a, b):
    """Matrix-matrix or matrix-vector product of two tensors.

    This wraps both mathvec and matmul into a single function, to mimic the
    behavior of torch's and numpy's versions of matmul
    """
    if ndim(b) < ndim(a) and (ndim(b) == 1 or b.shape[-2] != a.shape[-1]):
        return _tf.linalg.matvec(a, b)
    return _tf.linalg.matmul(a, b)


def outer(x, y):
    return _tf.einsum("i,j->ij", x, y)


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


def cast(x, dtype):
    return _tf.cast(x, dtype)


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


def dot(x, y):
    return _tf.tensordot(x, y, axes=1)


def isclose(x, y, rtol=rtol, atol=atol):
    if not _tf.is_tensor(x):
        x = _tf.constant(x)
    if not _tf.is_tensor(y):
        y = _tf.constant(y)
    x, y = convert_to_wider_dtype([x, y])
    dtype = x.dtype

    # rhs = _tf.constant(atol, dtype=dtype) + _tf.constant(rtol, dtype=dtype) * _tf.abs(y)
    diff = _tf.abs(_tf.subtract(x, y))
    rhs = _tf.constant(atol, dtype=dtype) + _tf.constant(rtol, dtype=dtype) * _tf.cast(
        _tf.abs(y), dtype=dtype
    )
    rhs = _tf.cast(rhs, dtype=diff.dtype)
    return _tf.less_equal(_tf.abs(_tf.subtract(x, y)), rhs)


def allclose(x, y, rtol=rtol, atol=atol):
    return _tf.reduce_all(isclose(x, y, rtol=rtol, atol=atol))


def eye(n, m=None):
    if m is None:
        m = n
    return _tf.eye(num_rows=n, num_columns=m)


def sum(x, axis=None, keepdims=False, name=None):
    if not _tf.is_tensor(x):
        x = _tf.convert_to_tensor(x)
    if x.dtype == bool:
        x = cast(x, int32)
    return _tf.reduce_sum(x, axis, keepdims, name)


def einsum(equation, *inputs, **kwargs):
    einsum_str = equation
    input_tensors_list = inputs

    input_tensors_list = convert_to_wider_dtype(input_tensors_list)

    einsum_list = einsum_str.split("->")
    input_str = einsum_list[0]
    output_str = einsum_list[1]

    input_str_list = input_str.split(",")

    is_ellipsis = [input_str[:3] == "..." for input_str in input_str_list]
    all_ellipsis = bool(_np.prod(is_ellipsis))

    if all_ellipsis:
        if len(input_str_list) > 2:
            raise NotImplementedError(
                "Ellipsis support not implemented for >2 input tensors"
            )
        ndims = [len(input_str[3:]) for input_str in input_str_list]

        tensor_a = input_tensors_list[0]
        tensor_b = input_tensors_list[1]
        initial_ndim_a = tensor_a.ndim
        initial_ndim_b = tensor_b.ndim
        tensor_a = to_ndarray(tensor_a, to_ndim=ndims[0] + 1)
        tensor_b = to_ndarray(tensor_b, to_ndim=ndims[1] + 1)

        n_tensor_a = tensor_a.shape[0]
        n_tensor_b = tensor_b.shape[0]

        if n_tensor_a != n_tensor_b:
            if n_tensor_a == 1:
                tensor_a = squeeze(tensor_a, axis=0)
                input_prefix_list = ["", "r"]
                output_prefix = "r"
            elif n_tensor_b == 1:
                tensor_b = squeeze(tensor_b, axis=0)
                input_prefix_list = ["r", ""]
                output_prefix = "r"
            else:
                raise ValueError("Shape mismatch for einsum.")
        else:
            input_prefix_list = ["r", "r"]
            output_prefix = "r"

        input_str_list = [
            input_str.replace("...", prefix)
            for input_str, prefix in zip(input_str_list, input_prefix_list)
        ]
        output_str = output_str.replace("...", output_prefix)

        input_str = input_str_list[0] + "," + input_str_list[1]
        einsum_str = input_str + "->" + output_str

        result = _tf.einsum(einsum_str, tensor_a, tensor_b, **kwargs)

        cond = (
            n_tensor_a == n_tensor_b == 1
            and initial_ndim_a != tensor_a.ndim
            and initial_ndim_b != tensor_b.ndim
        )

        if cond:
            result = squeeze(result, axis=0)
        return result

    return _tf.einsum(equation, *input_tensors_list, **kwargs)


def transpose(x, axes=None):
    return _tf.transpose(x, perm=axes)


def all(x, axis=None):
    return _tf.math.reduce_all(_tf.cast(x, bool), axis=axis)


def cumsum(a, axis=None):
    if axis is None:
        return _tf.math.cumsum(flatten(a), axis=0)
    return _tf.math.cumsum(a, axis=axis)


def cumprod(a, axis=None):
    if axis is None:
        return _tf.math.cumprod(flatten(a), axis=0)
    return _tf.math.cumprod(a, axis=axis)


# (sait) there is _tf.experimental.tril (we can use it once it moves to stable)
def tril(mat, k=0):
    if k not in (0, -1):
        raise NotImplementedError("Only k=0 and k=-1 supported so far")
    tril = _tf.linalg.band_part(mat, -1, 0)
    if k == 0:
        return tril
    zero_diag = _tf.zeros(mat.shape[:-1])
    return _tf.linalg.set_diag(tril, zero_diag)


# TODO(sait) use _tf.experimental.triu once it becomes stable.
def triu(mat, k=0):
    if k not in (0, 1):
        raise NotImplementedError("Only k=0 and k=1 supported so far")
    triu = _tf.linalg.band_part(mat, 0, -1)
    if k == 0:
        return triu
    zero_diag = _tf.zeros(mat.shape[:-1])
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
    if not _tf.is_tensor(x):
        x = _tf.constant(x)
    if not _tf.is_tensor(y):
        y = _tf.constant(y)
    y = cast(y, x.dtype)
    return _tf.where(condition, x, y)


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


def _vec_to_triu(vec):
    """Take vec and forms strictly upper triangular matrix.

    Parameters
    ----------
    vec : array_like, shape[..., n]

    Returns
    -------
    tril : array_like, shape=[..., k, k] where
        k is (1 + sqrt(1 + 8 * n)) / 2
    """
    n = vec.shape[-1]
    triu_shape = vec.shape + (n,)
    _ones = _tf.ones(triu_shape)
    vec = _tf.reshape(vec, [-1])
    mask_a = _tf.linalg.band_part(_ones, 0, -1)
    mask_b = _tf.linalg.band_part(_ones, 0, 0)
    mask = _tf.subtract(mask_a, mask_b)
    non_zero = _tf.not_equal(mask, _tf.constant(0.0))
    indices = _tf.where(non_zero)
    sparse = _tf.SparseTensor(indices, values=vec, dense_shape=triu_shape)
    return _tf.sparse.to_dense(sparse)


def _vec_to_tril(vec):
    """Take vec and forms strictly lower triangular matrix.

    Parameters
    ----------
    vec : array_like, shape=[..., n]

    Returns
    -------
    tril : array_like, shape=[..., k, k] where
        k is (1 + sqrt(1 + 8 * n)) / 2
    """
    n = vec.shape[-1]
    tril_shape = vec.shape + (n,)
    _ones = _tf.ones(tril_shape)
    vec = _tf.reshape(vec, [-1])
    mask_a = _tf.linalg.band_part(_ones, -1, 0)
    mask_b = _tf.linalg.band_part(_ones, 0, 0)
    mask = _tf.subtract(mask_a, mask_b)
    non_zero = _tf.not_equal(mask, _tf.constant(0.0))
    indices = _tf.where(non_zero)
    sparse = _tf.SparseTensor(indices, values=vec, dense_shape=tril_shape)
    return _tf.sparse.to_dense(sparse)


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
    triu_mat = _vec_to_triu(tri_upp)
    tril_mat = _vec_to_tril(tri_low)
    triu_tril_mat = triu_mat + tril_mat
    mat = _tf.linalg.set_diag(triu_tril_mat, diag)
    return mat


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


def linspace(*args, **kwargs):
    a = _tf.linspace(*args, **kwargs)
    if a.dtype is float64:
        a = cast(a, dtype=float32)

    return a
