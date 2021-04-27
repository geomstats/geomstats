"""Numpy based computation backend."""
import importlib
import os
import pkgutil
import sys
from importlib.util import module_from_spec, spec_from_file_location


def import_non_local(name: str):
    """Import non local module.

    Parameters
    ----------
        name (str): name of the module to import

    Returns
    -------
        [module]: the imported module if the import is succesfull
    """
    paths_hints = [p for p in sys.path if 'site-packages' in p]
    locations = [os.path.join(p, name + ".py") for p in paths_hints]
    locations += [os.path.join(p, name, "__init__.py") for p in paths_hints]
    spec = None

    for location in locations:
        spec = spec_from_file_location(name, location)
        if os.path.isfile(location) and spec:
            break

    if not spec:
        print(f"module not found in {locations}")
        return None

    the_module = module_from_spec(spec)
    sys.modules[spec.name] = the_module

    try:
        spec.loader.exec_module(the_module)
    except ModuleNotFoundError as e:
        print(f'error importing the module {name} with spec {spec} : {e}')
        return None

    return the_module


def import_non_local_submodule(main_module, submodule_name):
    """Import a non local submodule from a main module.

    Parameters
    ----------
        main_module: the main module
        submodule_name : the submodule to import

    Returns
    -------
        Module: the imported module or None
    """
    path = [os.path.dirname(main_module.__file__)]
    for _, module_name, is_pkg in pkgutil.walk_packages(path):
        if is_pkg and module_name == submodule_name:
            full_name = main_module.__name__ + '.' + module_name
            return importlib.import_module(full_name)
    return None

import autograd # NOQA

sys_autograd = import_non_local('autograd') # NOQA
np = import_non_local_submodule(sys_autograd, 'numpy')
abs = getattr(np, 'abs') # NOQA
all = getattr(np, 'all') # NOQA
allclose = getattr(np, 'allclose') # NOQA
amax = getattr(np, 'amax') # NOQA
amin = getattr(np, 'amin') # NOQA
any = getattr(np, 'any') # NOQA
arange = getattr(np, 'arange') # NOQA
arccos = getattr(np, 'arccos') # NOQA
arccosh = getattr(np, 'arccosh') # NOQA
arcsin = getattr(np, 'arcsin') # NOQA
arctan2 = getattr(np, 'arctan2') # NOQA
arctanh = getattr(np, 'arctanh') # NOQA
argmax = getattr(np, 'argmax') # NOQA
argmin = getattr(np, 'argmin') # NOQA
array = getattr(np, 'array') # NOQA
broadcast_arrays = getattr(np, 'broadcast_arrays') # NOQA
ceil = getattr(np, 'ceil') # NOQA
clip = getattr(np, 'clip') # NOQA
concatenate = getattr(np, 'concatenate') # NOQA
cos = getattr(np, 'cos') # NOQA
cosh = getattr(np, 'cosh') # NOQA
cross = getattr(np, 'cross') # NOQA
cumprod = getattr(np, 'cumprod') # NOQA
cumsum = getattr(np, 'cumsum') # NOQA
diagonal = getattr(np, 'diagonal') # NOQA
divide = getattr(np, 'divide') # NOQA
dot = getattr(np, 'dot') # NOQA
dtype = getattr(np, 'dtype') # NOQA
einsum = getattr(np, 'einsum') # NOQA
empty = getattr(np, 'empty') # NOQA
empty_like = getattr(np, 'empty_like') # NOQA
equal = getattr(np, 'equal') # NOQA
exp = getattr(np, 'exp') # NOQA
expand_dims = getattr(np, 'expand_dims') # NOQA
eye = getattr(np, 'eye') # NOQA
flip = getattr(np, 'flip') # NOQA
float32 = getattr(np, 'float32') # NOQA
float64 = getattr(np, 'float64') # NOQA
floor = getattr(np, 'floor') # NOQA
greater = getattr(np, 'greater') # NOQA
hsplit = getattr(np, 'hsplit') # NOQA
hstack = getattr(np, 'hstack') # NOQA
int32 = getattr(np, 'int32') # NOQA
int64 = getattr(np, 'int64') # NOQA
isclose = getattr(np, 'isclose') # NOQA
isnan = getattr(np, 'isnan') # NOQA
less = getattr(np, 'less') # NOQA
less_equal = getattr(np, 'less_equal') # NOQA
linspace = getattr(np, 'linspace') # NOQA
log = getattr(np, 'log') # NOQA
logical_and = getattr(np, 'logical_and') # NOQA
logical_or = getattr(np, 'logical_or')
matmul = getattr(np, 'matmul') # NOQA
maximum = getattr(np, 'maximum') # NOQA
mean = getattr(np, 'mean') # NOQA
meshgrid = getattr(np, 'meshgrid') # NOQA
mod = getattr(np, 'mod') # NOQA
ones = getattr(np, 'ones') # NOQA
ones_like = getattr(np, 'ones_like') # NOQA
outer = getattr(np, 'outer') # NOQA
power = getattr(np, 'power') # NOQA
repeat = getattr(np, 'repeat') # NOQA
reshape = getattr(np, 'reshape') # NOQA
shape = getattr(np, 'shape') # NOQA
sign = getattr(np, 'sign') # NOQA
sin = getattr(np, 'sin') # NOQA
sinh = getattr(np, 'sinh') # NOQA
split = getattr(np, 'split') # NOQA
sqrt = getattr(np, 'sqrt') # NOQA
squeeze = getattr(np, 'squeeze') # NOQA
stack = getattr(np, 'stack') # NOQA
std = getattr(np, 'std') # NOQA
sum = getattr(np, 'sum') # NOQA
tan = getattr(np, 'tan') # NOQA
tanh = getattr(np, 'tanh') # NOQA
tile = getattr(np, 'tile') # NOQA
trace = getattr(np, 'trace') # NOQA
transpose = getattr(np, 'transpose') # NOQA
triu_indices = getattr(np, 'triu_indices') # NOQA
tril_indices = getattr(np, 'tril_indices') # NOQA
searchsorted = getattr(np, 'searchsorted') # NOQA
tril = getattr(np, 'tril') # NOQA
uint8 = getattr(np, 'uint8') # NOQA
vstack = getattr(np, 'vstack') # NOQA
where = getattr(np, 'where') # NOQA
zeros = getattr(np, 'zeros') # NOQA
zeros_like = getattr(np, 'zeros_like') # NOQA

autograd_scipy = import_non_local_submodule(sys_autograd,
                                            'scipy.special')
polygamma = getattr(autograd_scipy, 'polygamma')

from scipy.sparse import coo_matrix # NOQA

from . import linalg  # NOQA
from . import random  # NOQA
from .common import to_ndarray  # NOQA
from ..constants import np_atol, np_rtol # NOQA

DTYPES = {
    dtype('int32'): 0,
    dtype('int64'): 1,
    dtype('float32'): 2,
    dtype('float64'): 3}


atol = np_atol
rtol = np_rtol


def to_numpy(x):
    return x


def convert_to_wider_dtype(tensor_list):
    dtype_list = [DTYPES[x.dtype] for x in tensor_list]
    wider_dtype_index = max(dtype_list)

    wider_dtype = list(DTYPES.keys())[wider_dtype_index]

    tensor_list = [cast(x, new_type=wider_dtype) for x in tensor_list]
    return tensor_list


def flatten(x):
    return x.flatten()


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


def _is_boolean(x):
    if isinstance(x, bool):
        return True
    if isinstance(x, (tuple, list)):
        return _is_boolean(x[0])
    if isinstance(x, np.ndarray):
        return x.dtype == bool
    return False


def _is_iterable(x):
    if isinstance(x, (list, tuple)):
        return True
    if isinstance(x, np.ndarray):
        return ndim(x) > 0
    return False


def assignment(x, values, indices, axis=0):
    """Assign values at given indices of an array.

    Parameters
    ----------
    x: array-like, shape=[dim]
        Initial array.
    values: {float, list(float)}
        Value or list of values to be assigned.
    indices: {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)

    use_vectorization = hasattr(indices, '__len__') and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] = values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    len_indices = len(indices) if _is_iterable(indices) else 1
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        if not zip_indices:
            len_indices = len(indices) if _is_iterable(indices) else 1
        len_values = len(values) if _is_iterable(values) else 1
        if len_values > 1 and len_values != len_indices:
            raise ValueError('Either one value or as many values as indices')
        x_new[indices] = values
    else:
        indices = tuple(
            list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] = values
    return x_new


def assignment_by_sum(x, values, indices, axis=0):
    """Add values at given indices of an array.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    values : {float, list(float)}
        Value or list of values to be assigned.
    indices : {int, tuple, list(int), list(tuple)}
        Single int or tuple, or list of ints or tuples of indices where value
        is assigned.
        If the length of the tuples is shorter than ndim(x), values are
        assigned to each copy along axis.
    axis: int, optional
        Axis along which values are assigned, if vectorized.

    Returns
    -------
    x_new : array-like, shape=[dim]
        Copy of x with the values assigned at the given indices.

    Notes
    -----
    If a single value is provided, it is assigned at all the indices.
    If a list is given, it must have the same length as indices.
    """
    x_new = copy(x)

    use_vectorization = hasattr(indices, '__len__') and len(indices) < ndim(x)
    if _is_boolean(indices):
        x_new[indices] += values
        return x_new
    zip_indices = _is_iterable(indices) and _is_iterable(indices[0])
    if zip_indices:
        indices = tuple(zip(*indices))
    if not use_vectorization:
        len_indices = len(indices) if _is_iterable(indices) else 1
        len_values = len(values) if _is_iterable(values) else 1
        if len_values > 1 and len_values != len_indices:
            raise ValueError('Either one value or as many values as indices')
        x_new[indices] += values
    else:
        indices = tuple(
            list(indices[:axis]) + [slice(None)] + list(indices[axis:]))
        x_new[indices] += values
    return x_new


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
    >>> a = np.array(range(30)).reshape(3,10)
    >>> get_slice(a, ((0, 2), (8, 9)))
    array([8, 29])
    """
    return x[indices]


def vectorize(x, pyfunc, multiple_args=False, signature=None, **kwargs):
    if multiple_args:
        return np.vectorize(pyfunc, signature=signature)(*x)
    return np.vectorize(pyfunc, signature=signature)(x)


def cast(x, new_type):
    return x.astype(new_type)


def set_diag(x, new_diag):
    """Set the diagonal along the last two axis.

    Parameters
    ----------
    x : array-like, shape=[dim]
        Initial array.
    new_diag : array-like, shape=[dim[-2]]
        Values to set on the diagonal.

    Returns
    -------
    None

    Notes
    -----
    This mimics tensorflow.linalg.set_diag(x, new_diag), when new_diag is a
    1-D array, but modifies x instead of creating a copy.
    """
    arr_shape = x.shape
    x[..., range(arr_shape[-2]), range(arr_shape[-1])] = new_diag
    return x


def ndim(x):
    return x.ndim


def copy(x):
    return x.copy()


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
    return array(
        coo_matrix((data, list(zip(*indices))), target_shape).todense())


def erf(x):
    cst_erf = 8.0 / (3.0 * np.pi) * (np.pi - 3.0) / (4.0 - np.pi)
    return \
        np.sign(x) * \
        np.sqrt(1 - np.exp(-x * x *
                           (4 / np.pi + cst_erf * x * x) /
                           (1 + cst_erf * x * x)))


def triu_to_vec(x, k=0):
    n = x.shape[-1]
    rows, cols = triu_indices(n, k=k)
    return x[..., rows, cols]
