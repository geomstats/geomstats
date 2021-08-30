"""Wrapper around autograd functions to be consistent with backends."""

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

        wrapped_function = function #primitive(function)
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

        # import funcsigs
        
        # sig = funcsigs.signature(wrapped_function)
        # print(sig.parameters)
        # bindings = sig.bind(arg_x, arg_y)
        # print(bindings.arguments)
        # print("bindings.arguments")

    return _value_and_grad(objective)
