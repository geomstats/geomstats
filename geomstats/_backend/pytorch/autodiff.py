"""Automatic differentiation in PyTorch."""

import functools

import torch as _torch
from torch.autograd.functional import hessian as _torch_hessian
from torch.autograd.functional import jacobian as _torch_jacobian


def _get_max_ndim_point(*points):
    """Identify point with higher dimension.

    Same as `geomstats.vectorization._get_max_ndim_point`.

    Parameters
    ----------
    points : array-like

    Returns
    -------
    max_ndim_point : array-like
        Point with higher dimension.
    """
    max_ndim_point = points[0]
    for point in points[1:]:
        if point.ndim > max_ndim_point.ndim:
            max_ndim_point = point

    return max_ndim_point


def _get_batch_shape(*points, point_ndims=1):
    """Get batch shape.

    Similar to `geomstats.vectorization.get_batch_shape`.

    Parameters
    ----------
    points : array-like or None
        Point belonging to the space.
    point_ndims : int or tuple[int]
        Point number of array dimensions.

    Returns
    -------
    batch_shape : tuple
        Returns the shape related with batch. () if only one point.
    """
    if isinstance(point_ndims, int):
        point_max_ndim = _get_max_ndim_point(*points)
        return point_max_ndim.shape[:-point_ndims]

    for point, point_ndim in zip(points, point_ndims):
        if point.ndim > point_ndim:
            return point.shape[:-point_ndim]

    return ()


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

        class FuncWithGrad(_torch.autograd.Function):
            """Wrapper class for a function with custom grad."""

            @staticmethod
            def forward(ctx, *args, **kwargs):
                ctx.save_for_backward(*args)
                return func(*args, **kwargs)

            @staticmethod
            def backward(ctx, grad_output):
                inputs = ctx.saved_tensors

                if grad_output.ndim > 0:
                    return tuple(
                        (
                            _torch.einsum(
                                "n,n...->n...", grad_output, custom_grad(*inputs)
                            )
                            if input_.requires_grad
                            else None
                        )
                        for input_, custom_grad in zip(inputs, grad_funcs)
                    )

                return tuple(
                    grad_output * custom_grad(*inputs) if input_.requires_grad else None
                    for input_, custom_grad in zip(inputs, grad_funcs)
                )

        def wrapped_function(*args, **kwargs):
            return FuncWithGrad.apply(*args, **kwargs)

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


def value_and_grad(func, argnums=0, point_ndims=1):
    """Return a function that returns func's value and gradients' values.

    Suitable for use in scipy.optimize.

    Parameters
    ----------
    func : callable
        Function whose value and gradient values
        will be computed. It must be real-valued.
    argnums: int or tuple[int]
        Specifies arguments to compute gradients with respect to.
    point_ndims: int or tuple[int]
        Specifies arguments ndim.

    Returns
    -------
    func_with_grad : callable
        Function that returns func's value and
        func's gradients' values at its inputs args.
    """
    argnums_ = (argnums,) if isinstance(argnums, int) else argnums

    def func_with_grad(*inputs, **kwargs):
        """Return func's value and func's gradients' values at args.

        Parameters
        ----------
        inputs : array-like, shape=[..., *point_shape]
        kwargs : dict
            Keyword arguments to function func and its gradients.

        Returns
        -------
        value : array-like, shape=[...,]
            Image of func at point.
        grad : array-like or tuple[array-like], shape=[..., *point_shape]
            Gradient of func at required points.
        """
        batch_shape = _get_batch_shape(*inputs, point_ndims=point_ndims)

        if len(inputs) > 1:
            point_ndims_ = (
                (point_ndims,) * len(inputs)
                if isinstance(point_ndims, int)
                else point_ndims
            )
            inputs_ = []
            for point, point_ndim in zip(inputs, point_ndims_):
                if point.shape[:-point_ndim] != batch_shape:
                    point = _torch.broadcast_to(point, batch_shape + point.shape)
                inputs_.append(point)
            inputs = inputs_

        inputs_ = []
        for index, point in enumerate(inputs):
            if index in argnums_ and not point.requires_grad:
                point = point.detach().requires_grad_(True)
            inputs_.append(point)
        inputs = inputs_
        value = func(*inputs, **kwargs).requires_grad_(True)

        if value.ndim > 0:
            sum_value = value.sum(axis=-1)
            sum_value.backward()
        else:
            value.backward()

        grads = tuple(
            point.grad.detach()
            for index, point in enumerate(inputs)
            if point.requires_grad and index in argnums_
        )
        if isinstance(argnums, int):
            grads = grads[0]
        return value.detach(), grads

    return func_with_grad


def value_and_jacobian(func):
    """Compute value and jacobian.

    NB: this is a naive implementation for consistency with autograd.

    Parameters
    ----------
    func : callable
        Function whose jacobian values will be computed.
    """

    def _value_and_jacobian(*args, **kwargs):
        """Return func's jacobian at args.

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
        """
        return (
            func(*args, **kwargs),
            jacobian_vec(func)(*args, **kwargs),
        )

    return _value_and_jacobian


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
