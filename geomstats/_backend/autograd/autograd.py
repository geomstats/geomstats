"""Wrapper around autograd functions to be consistent with backends."""

from autograd import jacobian as _jacobian
from autograd import value_and_grad as _value_and_grad


def value_and_grad(objective):
    """Wrap autograd value_and_grad function."""
    return _value_and_grad(objective)


def jacobian(f):
    """Wrap autograd jacobian function."""
    return _jacobian(f)
