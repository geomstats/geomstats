from autograd import elementwise_grad, jacobian, value_and_grad # NOQA
from autograd.extend import defjvp, defvjp, primitive


def value_and_grad(objective):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend. "
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend via the command "
        "export GEOMSTATS_BACKEND=autograd in a terminal.")


def jacobian(f):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend. "
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend via the command "
        "export GEOMSTATS_BACKEND=autograd in a terminal.")


def custom_gradient(*grad_func):
    """Decorate a function to define its custom gradient

    Parameters
    ----------
    *grad_func : ([callables]): Custom gradient functions
    """
    def decorator(function):

        wrapped_function = primitive(function)
        if len(grad_func) == 1:
            defvjp(
                wrapped_function,
                lambda ans, *args: lambda g: g * grad_func[0](ans, *args))

        return wrapped_function
    return decorator
