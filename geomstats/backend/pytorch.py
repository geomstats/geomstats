"""Pytorch based computation backend."""

import numpy as np
import torch

double = 'torch.DoubleTensor'
float16 = 'torch.Float'
float64 = 'torch.DoubleTensor'
int32 = 'torch.LongTensor'
int8 = 'torch.ByteTensor'


def arctan2(*args, **kwargs):
    return torch.arctan2(*args, **kwargs)


def cast(x, dtype):
    return x.type(dtype)


def divide(*args, **kwargs):
    return torch.div(*args, **kwargs)


def repeat(*args, **kwargs):
    return torch.repeat(*args, **kwargs)


def asarray(x):
    return np.asarray(x)


def concatenate(seq, axis=0, out=None):
    return torch.cat(seq, dim=axis, out=out)


def identity(val):
    return torch.eye(val)


def hstack(val):
    return torch.hstack(val)


def stack(*args, **kwargs):
    return torch.stack(*args, **kwargs)


def vstack(val):
    return torch.vstack(val)


def array(val):
    return torch.tensor(val)


def abs(val):
    return torch.abs(val)


def zeros(*args):
    return torch.from_numpy(np.zeros(*args)).float()


def ones(val):
    return torch.ones(val)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs)


def empty_like(*args, **kwargs):
    return torch.empty_like(*args, **kwargs)


def all(x, axis=None):
    if axis is None:
        return x.byte().all()
    return torch.from_numpy(np.all(x, axis=axis).astype(int))


def allclose(a, b, **kwargs):
    return torch.allclose(a, b, **kwargs)


def sin(val):
    return torch.sin(val)


def cos(val):
    return torch.cos(val)


def cosh(*args, **kwargs):
    return torch.cosh(*args, **kwargs)


def sinh(*args, **kwargs):
    return torch.sinh(*args, **kwargs)


def tanh(*args, **kwargs):
    return torch.tanh(*args, **kwargs)


def arcsinh(x):
    return torch.log(x + torch.sqrt(x*x+1))


def arcosh(x):
    return torch.log(x + torch.sqrt(x*x-1))


def tan(val):
    return torch.tan(val)


def arcsin(val):
    return torch.asin(val)


def arccos(val):
    return torch.acos(val)


def shape(val):
    return val.shape


def dot(a, b):
    return torch.from_numpy(np.dot(a, b))


def maximum(a, b):
    return torch.max(array(a), array(b))


def greater_equal(a, b):
    return torch.greater_equal(a, b)


def to_ndarray(x, to_ndim, axis=0):
    if type(x) == list:
        x = np.copy(np.array(x))
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    if x.dim() == to_ndim - 1:
        x = torch.unsqueeze(x, dim=axis)
    assert x.dim() >= to_ndim
    return x


def sqrt(val):
    return torch.sqrt(torch.tensor(val).float())


def norm(val, axis):
    return torch.linalg.norm(val, axis=axis)


def rand(*args, **largs):
    return torch.random.rand(*args, **largs)


def isclose(*args, **kwargs):
    return torch.from_numpy(np.isclose(*args, **kwargs).astype(int)).byte()


def less_equal(a, b):
    return np.less_equal(a, b)


def eye(*args, **kwargs):
    return torch.eye(*args, **kwargs)


def average(*args, **kwargs):
    return torch.average(*args, **kwargs)


def matmul(*args, **kwargs):
    return torch.matmul(*args, **kwargs)


def sum(*args, **kwargs):
    return torch.sum(*args, **kwargs)


def einsum(*args, **kwargs):
    return torch.from_numpy(np.einsum(*args, **kwargs)).float()


def T(x):
    return torch.t(x)


def transpose(x, axes=None):
    if axes:
        return x.permute(axes)
    if len(shape(x)) == 1:
        return x
    return x.t()


def squeeze(x, axis=None):
    return torch.squeeze(x, dim=axis)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs)


def trace(*args, **kwargs):
    return torch.from_numpy(np.trace(*args, **kwargs))


def mod(*args, **kwargs):
    return torch.fmod(*args, **kwargs)


def linspace(*args, **kwargs):
    return torch.linspace(*args, **kwargs)


def equal(*args, **kwargs):
    return torch.equal(*args, **kwargs)


def floor(*args, **kwargs):
    return torch.floor(*args, **kwargs)


def cross(x, y):
    return torch.from_numpy(np.cross(x, y))


def triu_indices(*args, **kwargs):
    return torch.triu_indices(*args, **kwargs)


def where(*args, **kwargs):
    return torch.where(*args, **kwargs)


def tile(x, y):
    # TODO(johmathe): Native tile implementation
    return array(np.tile(x, y))


def clip(x, amin, amax):
    return np.clip(x, amin, amax)


def diag(*args, **kwargs):
    return torch.diag(*args, **kwargs)


def any(x):
    return x.byte().any()


def expand_dims(x, axis):
    return torch.unsqueeze(x, dim=axis)


def outer(*args, **kwargs):
    return torch.ger(*args, **kwargs)


def hsplit(*args, **kwargs):
    return torch.hsplit(*args, **kwargs)


def argmax(*args, **kwargs):
    return torch.argmax(*args, **kwargs)


def diagonal(*args, **kwargs):
    return torch.diagonal(*args, **kwargs)


def exp(*args, **kwargs):
    return torch.exp(*args, **kwargs)


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


def copy(x):
    return x.clone()


def seed(x):
    torch.manual_seed(x)
