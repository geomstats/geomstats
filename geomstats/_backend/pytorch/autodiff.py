"""Automatic differentiation in PyTorch."""

import numpy as _np
import torch as _torch
from torch.autograd.functional import hessian as _torch_hessian
from torch.autograd.functional import jacobian as _torch_jacobian


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

        class func_with_grad(_torch.autograd.Function):
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

    We note that the jacobian function of torch is not vectorized
    by default, thus we modify its behavior here.

    Default pytorch behavior:

    If the jacobian for one point of shape (2,) is of shape (3, 2),
    then calling the jacobian on 4 points with shape (4, 2) will
    be of shape (3, 2, 4, 2).

    Modified behavior:

    Calling the jacobian on 4 points gives a tensor of shape (4, 3, 2).

    We use a for-loop to allow this function to be vectorized with
    respect to several inputs in point, because the flag vectorize=True
    fails.

    Parameters
    ----------
    func : callable
        Function whose jacobian is computed.

    Returns
    -------
    _ : callable
        Function taking point as input and returning
        the jacobian of func at point.
    """

    def _jacobian(point):
        if point.ndim == 1:
            return _torch_jacobian(func=lambda x: func(x), inputs=point)
        return _torch.stack(
            [
                _torch_jacobian(func=lambda x: func(x), inputs=one_point)
                for one_point in point
            ],
            axis=0,
        )

    return _jacobian


def hessian(func):
    """Return a function that returns the hessian of func.

    We modify the default behavior of the hessian function of torch
    to return a tensor of shape (n_points, dim, dim) when several
    points are given as inputs.

    Parameters
    ----------
    func : callable
        Function whose Hessian is computed.

    Returns
    -------
    _ : callable
        Function taking point as input and returning
        the hessian of func at point.
    """

    def _hessian(point):
        if point.ndim == 1:
            return _torch_hessian(func=lambda x: func(x), inputs=point, strict=True)
        return _torch.stack(
            [
                _torch_hessian(func=lambda x: func(x), inputs=one_point, strict=True)
                for one_point in point
            ],
            axis=0,
        )

    return _hessian


def jacobian_and_hessian(func):
    """Return a function that returns func's jacobian and hessian.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian
        will be computed. It must be real-valued.

    Returns
    -------
    func_with_jacobian_and_hessian : callable
        Function that returns func's jacobian and
        func's hessian at its inputs args.
    """

    def _jacobian_and_hessian(*args, **kwargs):
        """Return func's jacobian and func's hessian at args.

        Parameters
        ----------
        args : list
            Argument to function func and its gradients.
        kwargs : dict
            Keyword arguments to function func and its gradients.

        Returns
        -------
        jacobian : any
            Value of func's jacobian at input arguments args.
        hessian : any
            Value of func's hessian at input arguments args.
        """
        return jacobian(func)(*args), hessian(func)(*args)

    return _jacobian_and_hessian


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
                one_arg = _torch.from_numpy(_np.array(one_arg))
            if isinstance(one_arg, _np.ndarray):
                one_arg = _torch.from_numpy(one_arg)
            one_arg = one_arg.clone().requires_grad_(True)
            new_args = (*new_args, one_arg)
        args = new_args

        value = func(*args, **kwargs)
        if value.ndim > 0:
            value.backward(gradient=_torch.ones_like(one_arg), retain_graph=True)
        else:
            value.backward(retain_graph=True)

        all_grads = ()
        for one_arg in args:
            all_grads = (
                *all_grads,
                _torch.autograd.grad(value, one_arg, retain_graph=True)[0],
            )

        if to_numpy:
            value = detach(value).numpy()
            all_grads = [detach(one_grad).numpy() for one_grad in all_grads]

        if len(args) == 1:
            return value, all_grads[0]
        return value, all_grads

    return func_with_grad
