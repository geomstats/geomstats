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

    Returns
    -------
    x : array-like
        Detached tensor.
    """
    return x.detach()


def custom_gradient(*grad_funcs):
    """Create a decorator that allows a function to define its custom gradient(s).

    Parameters
    ----------
    *grad_funcs : callables
        Custom gradient functions.

    Returns
    -------
    decorator : callable
        This decorator, used on any function func, associates the
        input grad_funcs as the gradients of func.
    """

    def decorator(func):
        """Decorate a function to define its custome gradient(s).

        Parameters
        ----------
        func : callable
            Function whose gradients will be assigned by grad_funcs.

        Returns
        -------
        wrapped_function : callable
            Function func with gradients specified by grad_funcs.
        """

        class func_with_grad(torch.autograd.Function):
            """Wrapper class for a function with custom grad."""

            @staticmethod
            def forward(ctx, *args):
                ctx.save_for_backward(*args)
                return func(*args)

            @staticmethod
            def backward(ctx, grad_output):
                inputs = ctx.saved_tensors

                grads = ()
                for custom_grad in grad_funcs:
                    grads = (*grads, grad_output * custom_grad(*inputs))

                if len(grads) == 1:
                    return grads[0]
                return grads

        def wrapped_function(*args, **kwargs):
            new_inputs = args + tuple(kwargs.values())
            return func_with_grad.apply(*new_inputs)

        return wrapped_function

    return decorator


def jacobian(func):
    """Return a function that returns the jacobian of func.

    Parameters
    ----------
    func : callable
        Function whose Jacobian is computed.

    Returns
    -------
    _ : callable
        Function taking x as input and returning
        the jacobian of func at x.
    """
    return lambda x: torch_jac(func, x)


def value_and_grad(func, to_numpy=False):
    """Return a function that returns func's value and gradients' values.

    Suitable for use in scipy.optimize with to_numpy=True.

    Parameters
    ----------
    func : callable
        Function whose value and gradient values
        will be computed. It must be real-valued.
    to_numpy : bool
        Determines if the outputs value and grad will be cast
        to numpy arrays. Set to "True" when using scipy.optimize.
        Optional, default: False.

    Returns
    -------
    func_with_grad : callable
        Function that returns func's value and
        func's gradients' values at its inputs args.
    """

    def func_with_grad(*args, **kwargs):
        """Return func's value and func's gradients' values at args.

        Parameters
        ----------
        args : list
            Argument to function func and its gradients.
        kwargs : dict
            Keyword arguments to function func and its gradients.

        Returns
        -------
        value : any
            Value of func at input arguments args.
        all_grads : list or any
            Values of func's gradients at input arguments args.
        """
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
