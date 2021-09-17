"""Automatic differentiation in PyTorch."""

import numpy as np
import torch
from torch.autograd.functional import jacobian as torch_jac


def detach(x):
    """Return a new tensor detached from the current graph.

    Parameters
    ----------
    x : array-like
        Tensor to detach.
    """
    return x.detach()


def custom_gradient(*grad_func):
    """Decorate a function to define its custom gradient(s).

    Parameters
    ----------
    *grad_funcs : callables
        Custom gradient functions.
    """

    def decorator(func):
        class func_with_grad(torch.autograd.Function):
            """Wrapper for a function with custom grad."""

            @staticmethod
            def forward(ctx, *args):
                ctx.save_for_backward(*args)
                return func(*args)

            @staticmethod
            def backward(ctx, grad_output):
                inputs = ctx.saved_tensors

                grads = ()
                for custom_grad in grad_func:
                    grads = (*grads, grad_output * custom_grad(*inputs))

                if len(grads) == 1:
                    return grads[0]
                return grads

        def wrapper(*args, **kwargs):
            new_inputs = args + tuple(kwargs.values())
            return func_with_grad.apply(*new_inputs)

        return wrapper

    return decorator


def jacobian(func):
    """Return a function that returns the jacobian of a function."""
    return lambda x: torch_jac(func, x)


def value_and_grad(func, to_numpy=False):
    """Return a function that returns both value and gradient.

    Suitable for use in scipy.optimize

    Parameters
    ----------
    func : callable
        Function to compute the gradient. It must be real-valued.

    Returns
    -------
    func_with_grad : callable
        Function that takes the argument of the func function as input
        and returns both value and grad at the input.
    """

    def func_with_grad(*args, **kwargs):
        new_args = ()
        for one_arg in args:
            if isinstance(one_arg, float):
                one_arg = torch.from_numpy(np.array(one_arg))
            if isinstance(one_arg, np.ndarray):
                one_arg = torch.from_numpy(one_arg)
            one_arg = one_arg.clone().requires_grad_(True)
            new_args = (*new_args, one_arg)
        args = new_args

        value = func(*args, **kwargs)
        if value.ndim > 0:
            value.backward(gradient=torch.ones_like(one_arg), retain_graph=True)
        else:
            value.backward(retain_graph=True)

        all_grads = ()
        for one_arg in args:
            all_grads = (
                *all_grads,
                torch.autograd.grad(value, one_arg, retain_graph=True)[0],
            )

        if to_numpy:
            value = detach(value).numpy()
            all_grads = [detach(one_grad).numpy() for one_grad in all_grads]

        if len(args) == 1:
            return value, all_grads[0]
        return value, all_grads

    return func_with_grad
