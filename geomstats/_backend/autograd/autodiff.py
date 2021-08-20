"""Wrapper around autograd functions to be consistent with backends."""

from autograd import elementwise_grad as _elementwise_grad
from autograd import jacobian as _jacobian
from autograd import value_and_grad as _value_and_grad
from autograd.extend import defvjp, primitive


def elementwise_grad(f):
    """Wrap autograd elementwise_grad function."""
    return _elementwise_grad(f)


def custom_gradient(*grad_func):
    """Decorate a function to define its custom gradient.

    Parameters
    ----------
    *grad_func : callables
        Custom gradient functions.
    """
    def decorator(function):

        wrapped_function = primitive(function)
        if len(grad_func) == 1:
            defvjp(
                wrapped_function,
                lambda ans, *args: lambda g: g * grad_func[0](ans, *args))

        return wrapped_function
    return decorator


def jacobian(f):
    """Wrap autograd jacobian function."""
    return _jacobian(f)


def value_and_grad(objective):
    """Wrap autograd value_and_grad function."""
    return _value_and_grad(objective)
