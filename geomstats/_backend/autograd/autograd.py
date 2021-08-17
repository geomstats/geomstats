"""Wrapper around autograd functions to be consistent with backends."""


from autograd import value_and_grad as _value_and_grad
from autograd import jacobian as _jacobian

def value_and_grad(objective):
    """Return an error when using automatic differentiation with numpy."""
    return _value_and_grad(objective)


def jacobian(f):
    """Return an error when using automatic differentiation with numpy."""
    return _jacobian(f)