"""Pytorch based computation backend."""

import numpy as _np
import torch

from . import linalg  # NOQA
from . import random  # NOQA

double = 'torch.DoubleTensor'
float16 = 'torch.Float'
float32 = 'torch.FloatTensor'
float64 = 'torch.DoubleTensor'
int32 = 'torch.LongTensor'
int8 = 'torch.ByteTensor'


def while_loop(cond, body, loop_vars, maximum_iterations):
    iteration = 0
    while cond(*loop_vars):
        loop_vars = body(*loop_vars)
        iteration += 1
        if iteration >= maximum_iterations:
            break
    return loop_vars


def logical_or(x, y):
    return x or y


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def amax(x):
    return torch.max(x)


def amin(x):
    return torch.min(x)


def boolean_mask(x, mask):
    idx = _np.argwhere(_np.asarray(mask))
    return x[idx]


def arctan2(*args, **kwargs):
    return torch.atan2(*args, **kwargs)


def cast(x, dtype):
    x = array(x)
    return x.type(dtype)


def divide(*args, **kwargs):
    return torch.div(*args, **kwargs)


def repeat(a, repeats, axis=None):
    if torch.__version__ >= '1.1':
        return torch.repeat_interleave(a, repeats, axis)
    if(axis is None):
        axis = 0
    shape = list(a.shape)
    shape[axis] = shape[axis] * repeats
    return a.repeat(*shape)


def asarray(x):
    return _np.asarray(x)


def concatenate(seq, axis=0, out=None):
    seq = [cast(t, float32) for t in seq]
    return torch.cat(seq, dim=axis, out=out)


def identity(val):
    return torch.eye(val)


def hstack(seq):
    return concatenate(seq, axis=1)


def stack(*args, **kwargs):
    return torch.stack(*args, **kwargs)


def vstack(seq):
    return concatenate(seq)


def array(val):
    if isinstance(val, list):
        if not isinstance(val[0], torch.Tensor):
            val = _np.copy(_np.array(val))
        else:
            val = concatenate(val)

    if isinstance(val, bool):
        val = _np.array(val)
    if isinstance(val, _np.ndarray):
        if val.dtype == bool:
            val = torch.from_numpy(_np.array(val, dtype=_np.uint8))
        elif val.dtype == _np.float32 or val.dtype == _np.float64:
            val = torch.from_numpy(_np.array(val, dtype=_np.float64))
        else:
            val = torch.from_numpy(val)

    if not isinstance(val, torch.Tensor):
        val = torch.Tensor([val])
    if val.dtype == torch.float64:
        val = val.float()
    return val


def abs(val):
    return torch.abs(val)


def zeros(*args):
    return torch.from_numpy(_np.zeros(*args)).float()


def ones(*args):
    return torch.from_numpy(_np.ones(*args)).float()


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs)


def empty_like(*args, **kwargs):
    return torch.empty_like(*args, **kwargs)


def all(x, axis=None):
    if axis is None:
        return x.byte().all()
    return torch.from_numpy(_np.all(_np.array(x), axis=axis).astype(int))


def allclose(a, b, **kwargs):
    a = torch.tensor(a)
    b = torch.tensor(b)
    a = a.float()
    b = b.float()
    a = to_ndarray(a, to_ndim=1)
    b = to_ndarray(b, to_ndim=1)
    n_a = a.shape[0]
    n_b = b.shape[0]
    ndim = len(a.shape)
    if n_a > n_b:
        reps = (int(n_a / n_b),) + (ndim - 1) * (1,)
        b = tile(b, reps)
    elif n_a < n_b:
        reps = (int(n_b / n_a),) + (ndim - 1) * (1,)
        a = tile(a, reps)
    return torch.allclose(a, b, **kwargs)


def sin(val):
    return torch.sin(val)


def cos(val):
    return torch.cos(val)


def cosh(*args, **kwargs):
    return torch.cosh(*args, **kwargs)


def arccosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1


def sinh(*args, **kwargs):
    return torch.sinh(*args, **kwargs)


def tanh(*args, **kwargs):
    return torch.tanh(*args, **kwargs)


def arcsinh(x):
    return torch.log(x + torch.sqrt(x * x + 1))


def arcosh(x):
    return torch.log(x + torch.sqrt(x * x - 1))


def tan(val):
    return torch.tan(val)


def arcsin(val):
    return torch.asin(val)


def arccos(val):
    return torch.acos(val)


def shape(val):
    return val.shape


def dot(a, b):
    dot = _np.dot(a, b)
    return torch.from_numpy(_np.array(dot)).float()


def maximum(a, b):
    return torch.max(array(a), array(b))


def greater(a, b):
    return torch.gt(a, b)


def greater_equal(a, b):
    return torch.greater_equal(a, b)


def to_ndarray(x, to_ndim, axis=0):
    x = array(x)
    if x.dim() == to_ndim - 1:
        x = torch.unsqueeze(x, dim=axis)
    return x


def sqrt(val):
    return torch.sqrt(torch.tensor(val).float())


def norm(val, axis):
    return torch.norm(val, 2, axis)


if torch.__version__ >= '1.1':
    def isclose(*args, **kwargs):
        return torch.from_numpy(_np.isclose(*args, **kwargs))
else:
    def isclose(*args, **kwargs):
        return torch.from_numpy(_np.isclose(*args, **kwargs).astype(_np.uint8))


def less(a, b):
    return torch.le(a, b)


def less_equal(a, b):
    return torch.le(a, b)


def eye(*args, **kwargs):
    return torch.eye(*args, **kwargs)


def average(*args, **kwargs):
    return torch.average(*args, **kwargs)


def matmul(*args, **kwargs):
    return torch.matmul(*args, **kwargs)


def sum(x, axis=None, keepdims=None, **kwargs):
    if axis is None:
        if keepdims is None:
            return torch.sum(x, **kwargs)
        return torch.sum(x, keepdim=keepdims, **kwargs)
    if keepdims is None:
        return torch.sum(x, dim=axis, **kwargs)
    return torch.sum(x, dim=axis, keepdim=keepdims, **kwargs)


def einsum(*args, **kwargs):
    return torch.from_numpy(_np.einsum(*args, **kwargs)).float()


def T(x):
    return torch.t(x)


def transpose(x, axes=None):
    if axes:
        return x.permute(axes)
    if len(shape(x)) == 1:
        return x
    return x.t()


def squeeze(x, axis=None):
    if axis is None:
        return torch.squeeze(x)
    else:
        return torch.squeeze(x, axis)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs)


def trace(*args, **kwargs):
    trace = _np.trace(*args, **kwargs)
    return torch.from_numpy(_np.array(trace)).float()


def mod(*args, **kwargs):
    return torch.fmod(*args, **kwargs)


def arctanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def linspace(start, stop, num):
    return torch.linspace(start=start, end=stop, steps=num)


def equal(a, b, **kwargs):
    if a.dtype == torch.ByteTensor:
        a = cast(a, torch.uint8).float()
    if b.dtype == torch.ByteTensor:
        b = cast(b, torch.uint8).float()
    return torch.eq(a, b, **kwargs)


def floor(*args, **kwargs):
    return torch.floor(*args, **kwargs)


def cross(x, y):
    return torch.from_numpy(_np.cross(x, y))


def triu_indices(*args, **kwargs):
    return torch.triu_indices(*args, **kwargs)


def where(test, x, y):
    return torch.where(test, torch.tensor(x), torch.tensor(y))


def tile(x, y):
    # TODO(johmathe): Native tile implementation
    return array(_np.tile(x, y))


def clip(x, amin, amax):
    if x.dtype == 'torch.float':
        return torch.clamp(x, amin, amax)
    return _np.clip(x, amin, amax)


def clamp(*args, **kwargs):
    return torch.clamp(*args, **kwargs)


def diag(*args, **kwargs):
    return torch.diag(*args, **kwargs)


def any(x):
    return x.byte().any()


def expand_dims(x, axis=0):
    return torch.unsqueeze(x, dim=axis)


def outer(*args, **kwargs):
    return torch.ger(*args, **kwargs)


def hsplit(*args, **kwargs):
    return torch.hsplit(*args, **kwargs)


def argmax(*args, **kwargs):
    return torch.argmax(*args, **kwargs)


def diagonal(*args, **kwargs):
    return torch.diagonal(*args, **kwargs)


def exp(input):
    return torch.exp(input)


def log(*args, **kwargs):
    return torch.log(*args, **kwargs)


def cov(*args, **kwargs):
    return torch.cov(*args, **kwargs)


def eval(x):
    return x


def ndim(x):
    return x.dim()


def gt(*args, **kwargs):
    return torch.gt(*args, **kwargs)


def eq(*args, **kwargs):
    return torch.eq(*args, **kwargs)


def nonzero(*args, **kwargs):
    return torch.nonzero(*args, **kwargs)


def seed(x):
    torch.manual_seed(x)


def prod(x, axis=None):
    if axis is None:
        return torch.prod(x)
    else:
        return torch.prod(x, dim=axis)


def sign(*args, **kwargs):
    return torch.sign(*args, **kwargs)


def mean(x, axis=None):
    if axis is None:
        return torch.mean(x)
    else:
        return _np.mean(x, axis)


def argmin(*args, **kwargs):
    return torch.argmin(*args, **kwargs)


def arange(*args, **kwargs):
    return torch.arange(*args, **kwargs)


def gather(x, indices, axis=0):
    return x[indices]


def get_mask_i_float(i, n):
    range_n = arange(cast(array(n), int32)[0])
    i_float = cast(array(i), int32)
    mask_i = equal(range_n, i_float)
    mask_i_float = cast(mask_i, float32)
    return mask_i_float


def copy(x):
    return x.clone()


def cumprod(x, axis=0):
    if axis is None:
        raise NotImplementedError('cumprod is not defined where axis is None')
    else:
        return torch.cumprod(x, dim=axis)


def isnan(*args, **kwargs):
    return torch.isnan(*args, **kwargs)
