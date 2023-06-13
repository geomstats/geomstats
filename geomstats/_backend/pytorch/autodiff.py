"""Automatic differentiation in PyTorch."""

import functools

import numpy as _np
import torch as _torch
from torch.autograd.functional import hessian as _torch_hessian
from torch.autograd.functional import jacobian as _torch_jacobian


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
        return _torch_jacobian(func=lambda x: func(x), inputs=point)

    return _jacobian


def jacobian_vec(func, point_ndim=1):
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
        if point.ndim == point_ndim:
            return _torch_jacobian(func=lambda x: func(x), inputs=point)
        return _torch.stack(
            [
                _torch_jacobian(func=lambda x: func(x), inputs=one_point)
                for one_point in point
            ],
            axis=0,
        )

    return _jacobian


def hessian(func, func_out_ndim=0):
    """Return a function that returns the hessian of func.

    Parameters
    ----------
    func : callable
        Function whose Hessian is computed.
    func_out_ndim : dim
        func output ndim.

    Returns
    -------
    _ : callable
        Function taking point as input and returning
        the hessian of func at point.
    """

    def _hessian(point):
        return _torch_hessian(func=lambda x: func(x), inputs=point, strict=True)

    def _hessian_vector_valued(point):
        def scalar_func(point, a):
            return func(point)[a]

        return _torch.stack(
            [
                hessian(functools.partial(scalar_func, a=a))(point)
                for a in range(func_out_ndim)
            ]
        )

    if func_out_ndim:
        return _hessian_vector_valued

    return _hessian


def hessian_vec(func, point_ndim=1, func_out_ndim=0):
    """Return a function that returns the hessian of func.

    We modify the default behavior of the hessian function of torch
    to return a tensor of shape (n_points, dim, dim) when several
    points are given as inputs.

    Parameters
    ----------
    func : callable
        Function whose Hessian is computed.
    func_out_ndim : dim
        func output ndim.

    Returns
    -------
    _ : callable
        Function taking point as input and returning
        the hessian of func at point.
    """
    hessian_func = hessian(func, func_out_ndim=func_out_ndim)

    def _hessian(point):
        if point.ndim == point_ndim:
            return hessian_func(point)
        return _torch.stack(
            [hessian_func(one_point) for one_point in point],
            axis=0,
        )

    return _hessian


def jacobian_and_hessian(func, func_out_ndim=0):
    """Return a function that returns func's jacobian and hessian.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian
        will be computed. It must be real-valued.
    func_out_ndim : dim
        func output ndim.

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
        return jacobian(func)(*args), hessian(func, func_out_ndim=func_out_ndim)(*args)

    return _jacobian_and_hessian


def value_and_grad(func, argnums=0, to_numpy=False):
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
    if isinstance(argnums, int):
        argnums = (argnums,)

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
        new_args = []
        for i_arg, one_arg in enumerate(args):
            if isinstance(one_arg, float):
                one_arg = _torch.from_numpy(_np.array(one_arg))
            if isinstance(one_arg, _np.ndarray):
                one_arg = _torch.from_numpy(one_arg)

            requires_grad = i_arg in argnums
            one_arg = one_arg.detach().requires_grad_(requires_grad)
            new_args.append(one_arg)

        value = func(*new_args, **kwargs)
        value = value.requires_grad_(True)

        if value.ndim > 0:
            sum_value = value.sum()
            sum_value.backward()
        else:
            value.backward()

        all_grads = []
        for i_arg, one_arg in enumerate(new_args):
            if i_arg in argnums:
                all_grads.append(
                    one_arg.grad,
                )

        if to_numpy:
            value = value.detach().numpy()
            all_grads = [one_grad.detach().numpy() for one_grad in all_grads]

        if len(new_args) == 1:
            return value, all_grads[0]
        return value, tuple(all_grads)

    return func_with_grad


def value_jacobian_and_hessian(func, func_out_ndim=0):
    """Compute value, jacobian and hessian.

    func is called as many times as the output dim.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian values
        will be computed.
    func_out_ndim : int
        func output ndim.
    """

    def _value_jacobian_and_hessian(*args, **kwargs):
        """Return func's jacobian and func's hessian at args.

        Parameters
        ----------
        args : list
            Argument to function func and its gradients.
        kwargs : dict
            Keyword arguments to function func and its gradients.

        Returns
        -------
        value : array-like
            Value of func at input arguments args.
        jacobian : array-like
            Value of func's jacobian at input arguments args.
        hessian : array-like
            Value of func's hessian at input arguments args.
        """
        return (
            func(*args, **kwargs),
            jacobian_vec(func)(*args, **kwargs),
            hessian_vec(func, func_out_ndim=func_out_ndim)(*args, **kwargs),
        )

    return _value_jacobian_and_hessian
