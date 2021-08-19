from autograd import elementwise_grad, jacobian, value_and_grad # NOQA
from autograd.extend import defjvp, defvjp, primitive


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
