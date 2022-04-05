"""Placeholders with error messages.

NumPy backend does not offer automatic differentiation.
The following functions return error messages.
"""


def detach(x):
    """Return a new tensor detached from the current graph.

    This is a placeholder in order to have consistent backend APIs.

    Parameters
    ----------
    x : array-like
        Tensor to detach.
    """
    return x


def value_and_grad(*args, **kwargs):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend. "
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend via the command "
        "export GEOMSTATS_BACKEND=autograd in a terminal."
    )


def jacobian(func):
    """Return an error when using automatic differentiation with numpy."""
    raise RuntimeError(
        "Automatic differentiation is not supported with numpy backend. "
        "Use autograd, pytorch or tensorflow backend instead.\n"
        "Change backend via the command "
        "export GEOMSTATS_BACKEND=autograd in a terminal."
    )


def custom_gradient(*grad_funcs):
    """Decorate a function to define its custom gradient(s).

    This is a placeholder in order to have consistent backend APIs.
    """

    def decorator(func):
        return func

    return decorator
