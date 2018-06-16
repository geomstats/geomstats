"""Pytorch based computation backend."""

import torch
import numpy as np

def arctan2(*args, **kwargs):
    return torch.arctan2(*args, **kwargs)


def cast(x, dtype):
    return x.astype(dtype)


def divide(*args, **kwargs):
    return torch.divide(*args, **kwargs)


def repeat(*args, **kwargs):
    return torch.repeat(*args, **kwargs)


def asarray(*args, **kwargs):
    return torch.asarray(*args, **kwargs)


def concatenate(*args, **kwargs):
    return torch.concatenate(*args, **kwargs)


def identity(val):
    return torch.identity(val)


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


def zeros(val):
    return torch.zeros(val)


def ones(val):
    return torch.ones(val)


def ones_like(*args, **kwargs):
    return torch.ones_like(*args, **kwargs)


def empty_like(*args, **kwargs):
    return torch.empty_like(*args, **kwargs)


def all(*args, **kwargs):
    return torch.all(*args, **kwargs)


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


def arccosh(*args, **kwargs):
    return torch.arccosh(*args, **kwargs)


def tan(val):
    return torch.tan(val)


def arcsin(val):
    return torch.arcsin(val)


def arccos(val):
    return torch.arccos(val)


def shape(val):
    return val.shape


def dot(a, b):
    return torch.dot(a, b)


def maximum(a, b):
    return torch.maximum(a, b)


def greater_equal(a, b):
    return torch.greater_equal(a, b)


def to_ndarray(x, to_ndim, axis=0):
    if x.dim() == to_ndim - 1:
        x = torch.unsqueeze(x, dim=axis)
    assert x.dim() >= to_ndim
    return x


def sqrt(val):
    return torch.sqrt(torch.tensor(val).double())


def norm(val, axis):
    return torch.linalg.norm(val, axis=axis)


def rand(*args, **largs):
    return torch.random.rand(*args, **largs)


def isclose(*args, **kwargs):
    print(array(args[1]))
    return torch.isclose(args[0], array(args[1]), *args[2:], **kwargs)


def less_equal(a, b):
    return torch.less_equal(a, b)


def eye(*args, **kwargs):
    return torch.eye(*args, **kwargs)


def average(*args, **kwargs):
    return torch.average(*args, **kwargs)


def matmul(*args, **kwargs):
    return torch.matmul(*args, **kwargs)


def sum(*args, **kwargs):
    return torch.sum(*args, **kwargs)


def einsum(*args, **kwargs):
    return torch.einsum(*args, **kwargs)


def transpose(*args, **kwargs):
    return torch.transpose(*args, **kwargs)


def squeeze(*args, **kwargs):
    return torch.squeeze(*args, **kwargs)


def zeros_like(*args, **kwargs):
    return torch.zeros_like(*args, **kwargs)


def trace(*args, **kwargs):
    return torch.trace(*args, **kwargs)


def mod(*args, **kwargs):
    return torch.mod(*args, **kwargs)


def linspace(*args, **kwargs):
    return torch.linspace(*args, **kwargs)


def equal(*args, **kwargs):
    return torch.equal(*args, **kwargs)


def floor(*args, **kwargs):
    return torch.floor(*args, **kwargs)


def cross(*args, **kwargs):
    print(args[1])
    return torch.cross(*args, **kwargs)


def triu_indices(*args, **kwargs):
    return torch.triu_indices(*args, **kwargs)


def where(*args, **kwargs):
    return torch.where(*args, **kwargs)


def tile(*args, **kwargs):
    # TODO(johmathe): Native tile implementation
    print(np.tile(*args, **kwargs))
    print("ME")
    return array(np.tile(*args, **kwargs))


def clip(*args, **kwargs):
    return torch.clip(*args, **kwargs)


def diag(*args, **kwargs):
    return torch.diag(*args, **kwargs)


def any(*args, **kwargs):
    return torch.any(*args, **kwargs)


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
