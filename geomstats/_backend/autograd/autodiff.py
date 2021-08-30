"""Wrapper around autograd functions to be consistent with backends."""

import funcsigs
from autograd import multigrad_dict
from autograd import numpy as np


from autograd import elementwise_grad as _elementwise_grad
from autograd import jacobian as _jacobian
from autograd import value_and_grad as _value_and_grad
from autograd.extend import defvjp, primitive


def detach(x):
    return x


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
                lambda ans, *args: lambda g: g * grad_func[0](*args))

        else:
            print(f"Number of grad functions: {len(grad_func)}")
            # vjps = []
            # for one_grad_func in grad_func:
            #     one_vjp = \
            #         lambda ans, *args: lambda g: g * one_grad_func(
            #             *args)
            #     vjps.append(one_vjp)
            # vjps = tuple(vjps)

            print(grad_func[0])
            print(grad_func[1])
            defvjp(
                wrapped_function, 
                lambda ans, *args: lambda g: g * grad_func[0](*args),
                lambda ans, *args: lambda g: g * grad_func[1](*args))

        return wrapped_function
    return decorator


def jacobian(f):
    """Wrap autograd jacobian function."""
    return _jacobian(f)


def value_and_grad(objective, to_numpy=False):
    """Wrap autograd value_and_grad function."""
    if "_is_autograd_primitive" in objective.__dict__:
        multigradfunc_dict = multigrad_dict(objective.fun)
    else:
        multigradfunc_dict = multigrad_dict(objective)
    def multigrad_val(*args):
        multigradvals_dict = multigradfunc_dict(*args)
        multigradvals = tuple(multigradvals_dict.values())
        multigradvals = multigradvals[0] if len(multigradvals) == 1 else multigradvals
        return multigradvals
    return lambda *args: (objective(*args), multigrad_val(*args))
