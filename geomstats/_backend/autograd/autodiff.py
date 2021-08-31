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
        wrapped_function = primitive(function)
        if len(grad_func) == 1:
            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: lambda g: g * grad_func[0](*args, **kwargs))
        elif len(grad_func) == 2:
            defvjp(
                wrapped_function, 
                lambda ans, *args, **kwargs: lambda g: g * grad_func[0](*args, **kwargs),
                lambda ans, *args, **kwargs: lambda g: g * grad_func[1](*args, **kwargs))
        elif len(grad_func) == 3:
            defvjp(
                wrapped_function, 
                lambda ans, *args, **kwargs: lambda g: g * grad_func[0](*args, **kwargs),
                lambda ans, *args, **kwargs: lambda g: g * grad_func[1](*args, **kwargs),
                lambda ans, *args, **kwargs: lambda g: g * grad_func[2](*args, **kwargs))
        else:
            raise NotImplementedError(
                "custom_gradient is not yet implemented for more than 3 gradients."
            )

        return wrapped_function
    return decorator


def jacobian(f):
    """Wrap autograd jacobian function."""
    return _jacobian(f)


def value_and_grad(objective, to_numpy=False):
    """Wrap autograd value_and_grad function."""

    def aux_value_and_grad(*args):
        n_args = len(args)
        value = objective(*args)

        all_grads = []
        for i in range(n_args):
            def objective_of_i(*args):
                reorg_args =  args[1:i+1] + (args[0],) + args[i+1:]
                return objective(*reorg_args)
            new_args = (args[i],) + args[:i] + args[i+1:]
            _, grad_i = _value_and_grad(objective_of_i)(*new_args)
            all_grads.append(grad_i)

        if n_args == 1:
            return value, all_grads[0]
        return value, tuple(all_grads)   

    return aux_value_and_grad