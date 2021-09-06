"""Wrapper around autograd functions to be consistent with backends."""


from autograd import elementwise_grad as _elementwise_grad
from autograd import jacobian as _jacobian
from autograd import value_and_grad as _value_and_grad
from autograd.extend import defvjp, primitive


def detach(x):
    """Return a new tensor detached from the current graph.

    This is a placeholder in order to have consistent backend APIs.

    Parameters
    ----------
    x : array-like
        Tensor to detach.
    """
    return x


def elementwise_grad(func):
    """Wrap autograd elementwise_grad function.

    Parameters
    ----------
    func : callable
        Function for which the element-wise grad is computed.
    """
    return _elementwise_grad(func)


def custom_gradient(*grad_funcs):
    """Decorate a function to define its custom gradient(s).

    Parameters
    ----------
    *grad_funcs : callables
        Custom gradient functions.
    """

    def decorator(func):
        wrapped_function = primitive(func)
        if len(grad_funcs) == 1:
            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: lambda g: g
                * grad_funcs[0](*args, **kwargs),
            )
        elif len(grad_funcs) == 2:
            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: lambda g: g
                * grad_funcs[0](*args, **kwargs),
                lambda ans, *args, **kwargs: lambda g: g
                * grad_funcs[1](*args, **kwargs),
            )
        elif len(grad_funcs) == 3:
            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: lambda g: g
                * grad_funcs[0](*args, **kwargs),
                lambda ans, *args, **kwargs: lambda g: g
                * grad_funcs[1](*args, **kwargs),
                lambda ans, *args, **kwargs: lambda g: g
                * grad_funcs[2](*args, **kwargs),
            )
        else:
            raise NotImplementedError(
                "custom_gradient is not yet implemented " "for more than 3 gradients."
            )

        return wrapped_function

    return decorator


def jacobian(func):
    """Wrap autograd jacobian function."""
    return _jacobian(func)


def value_and_grad(func, to_numpy=False):
    """Wrap autograd value_and_grad function."""

    def aux_value_and_grad(*args):
        n_args = len(args)
        value = func(*args)

        all_grads = []
        for i in range(n_args):

            def func_of_ith(*args):
                reorg_args = args[1 : i + 1] + (args[0],) + args[i + 1 :]
                return func(*reorg_args)

            new_args = (args[i],) + args[:i] + args[i + 1 :]
            _, grad_i = _value_and_grad(func_of_ith)(*new_args)
            all_grads.append(grad_i)

        if n_args == 1:
            return value, all_grads[0]
        return value, tuple(all_grads)

    return aux_value_and_grad
