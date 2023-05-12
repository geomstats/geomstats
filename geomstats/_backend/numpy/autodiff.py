"""Placeholders with error messages.

NumPy backend does not offer automatic differentiation.
The following functions return error messages.
"""

from geomstats.exceptions import AutodiffNotImplementedError

_USE_OTHER_BACKEND_MSG = (
    "Automatic differentiation is not supported with numpy backend. "
    "Use autograd or pytorch backend instead.\n"
    "Change backend via the command "
    "export GEOMSTATS_BACKEND=autograd in a terminal."
)


def value_and_grad(*args, **kwargs):
    """Return an error when using automatic differentiation with numpy."""
    raise AutodiffNotImplementedError(_USE_OTHER_BACKEND_MSG)


def jacobian(*args, **kwargs):
    """Return an error when using automatic differentiation with numpy."""
    raise AutodiffNotImplementedError(_USE_OTHER_BACKEND_MSG)


def jacobian_vec(*args, **kwargs):
    """Return an error when using automatic differentiation with numpy."""
    raise AutodiffNotImplementedError(_USE_OTHER_BACKEND_MSG)


def hessian(*args, **kwargs):
    """Return an error when using automatic differentiation with numpy."""
    raise AutodiffNotImplementedError(_USE_OTHER_BACKEND_MSG)


def hessian_vec(*args, **kwargs):
    """Return an error when using automatic differentiation with numpy."""
    raise AutodiffNotImplementedError(_USE_OTHER_BACKEND_MSG)


def jacobian_and_hessian(*args, **kwargs):
    """Return an error when using automatic differentiation with numpy."""
    raise AutodiffNotImplementedError(_USE_OTHER_BACKEND_MSG)


def custom_gradient(*grad_funcs):
    """Decorate a function to define its custom gradient(s).

    This is a placeholder in order to have consistent backend APIs.
    """

    def decorator(func):
        return func

    return decorator


def value_jacobian_and_hessian(*args, **kwargs):
    raise AutodiffNotImplementedError(_USE_OTHER_BACKEND_MSG)
