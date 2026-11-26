"""Automatic differentiation in PyTorch."""

import functools

import torch as _torch


def _broadcast_to_batch(inputs, point_ndims):
    """Broadcast single inputs to the batch shape of the batched ones.

    Parameters
    ----------
    inputs : list[array-like]
        Inputs, of shape ``[..., *point_shape]``.
    point_ndims : int or tuple[int]
        Number of axes of a point of each input.

    Returns
    -------
    inputs : list[array-like]
        Inputs, all with the same batch shape.
    """
    if isinstance(point_ndims, int):
        point_ndims = (point_ndims,) * len(inputs)

    batch_shape = _torch.broadcast_shapes(
        *(
            input_.shape[: input_.ndim - point_ndim]
            for input_, point_ndim in zip(inputs, point_ndims)
        )
    )
    return [
        _torch.broadcast_to(
            input_, batch_shape + input_.shape[input_.ndim - point_ndim :]
        )
        for input_, point_ndim in zip(inputs, point_ndims)
    ]


def _expand_and_multiply(grad_output, grad):
    """Multiply the output gradient by a gradient with extra point axes.

    Parameters
    ----------
    grad_output : array-like, shape=[...]
        Gradient with respect to the output, one value per point.
    grad : array-like, shape=[..., *point_shape]
        Gradient of the output with respect to an input.

    Returns
    -------
    vjp : array-like, shape=[..., *point_shape]
        Vector-jacobian product.
    """
    point_ndim = grad.ndim - grad_output.ndim
    return grad_output.reshape(grad_output.shape + (1,) * point_ndim) * grad


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
                """Compute func, saving its inputs for the backward pass."""
                ctx.save_for_backward(*args)
                return func(*args, **kwargs)

            @staticmethod
            def backward(ctx, grad_output):
                """Compute the gradients of the inputs with the custom gradients."""
                inputs = ctx.saved_tensors
                return tuple(
                    (
                        _expand_and_multiply(grad_output, custom_grad(*inputs))
                        if input_.requires_grad
                        else None
                    )
                    for input_, custom_grad in zip(inputs, grad_funcs)
                )

        @functools.wraps(func)
        def wrapped_function(*args, **kwargs):
            """Compute func, with gradients given by the custom gradients."""
            return FuncWithGrad.apply(*args, **kwargs)

        return wrapped_function

    return decorator


def _apply_pointwise(func_at_point, point, point_ndim):
    """Apply a function of a single point to each point of a batch.

    Parameters
    ----------
    func_at_point : callable
        Function of a single point, returning an array or a tuple of arrays.
    point : array-like, shape=[..., *point_shape]
        Point or batch of points.
    point_ndim : int
        Number of axes of a point. All preceding axes are batch axes.

    Returns
    -------
    result : array-like or tuple[array-like]
        Results stacked along the batch axes.
    """
    if point.ndim == point_ndim:
        return func_at_point(point)

    batch_ndim = point.ndim - point_ndim
    batch_shape = point.shape[:batch_ndim]
    points = point.reshape((-1,) + point.shape[batch_ndim:])
    results = [func_at_point(one_point) for one_point in points]

    def _stack(arrays):
        """Stack arrays along the batch axes."""
        stacked = _torch.stack(arrays)
        return stacked.reshape(batch_shape + stacked.shape[1:])

    if isinstance(results[0], tuple):
        return tuple(_stack(list(arrays)) for arrays in zip(*results))

    return _stack(results)


def _value_and_jacobian_at_point(func, point, has_aux=False):
    """Compute the value and the jacobian of func at a single point.

    func is evaluated once: the jacobian is computed row by row by
    backpropagating each output component through the same graph.

    The graph of the jacobian is built only when point requires grad.
    This is the case when the jacobian is itself being differentiated,
    since the inputs being differentiated are made to require grad.
    Nested derivatives, e.g. a jacobian of a jacobian, are therefore
    correct, while calls on a point that does not require grad return
    detached tensors.

    Parameters
    ----------
    func : callable
        Function of a single point. If ``has_aux``, it returns a tuple
        made of the output to differentiate and an auxiliary output.
    point : array-like, shape=[*point_shape]
        Point.
    has_aux : bool
        Whether func also returns an auxiliary output.

    Returns
    -------
    value : array-like, shape=[*out_shape]
        Value of func at point.
    jacobian : array-like, shape=[*out_shape, *point_shape]
        Jacobian of func at point.
    aux : any
        Auxiliary output of func. Only if ``has_aux``.
    """
    create_graph = point.requires_grad
    if not create_graph:
        point = point.detach().requires_grad_(True)

    value = func(point)
    if has_aux:
        value, aux = value

    rows = []
    for component in value.reshape(-1):
        row = None
        if component.requires_grad:
            (row,) = _torch.autograd.grad(
                component,
                point,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
        rows.append(_torch.zeros_like(point) if row is None else row)

    jac = (
        _torch.stack(rows).reshape(value.shape + point.shape)
        if rows
        else _torch.zeros(value.shape + point.shape, dtype=point.dtype)
    )

    if not create_graph:
        value, jac = value.detach(), jac.detach()
        if has_aux:
            aux = aux.detach()

    if has_aux:
        return value, jac, aux
    return value, jac


def _jacobian_at_point(func, point):
    """Compute the jacobian of func at a single point."""
    return _value_and_jacobian_at_point(func, point)[1]


def jacobian(func, point_ndim=1):
    """Return a function that returns the pointwise jacobian of func.

    Contrary to the jacobian of torch, the jacobian of a batch of points is
    the batch of the jacobians at each point: if the jacobian at a point of
    shape (2,) has shape (3, 2), then the jacobian at 4 points of shape
    (4, 2) has shape (4, 3, 2), and not (3, 2, 4, 2).

    Parameters
    ----------
    func : callable
        Function whose jacobian is computed.
    point_ndim : int
        Number of axes of a point. All preceding axes are batch axes.

    Returns
    -------
    _ : callable
        Function taking point as input and returning
        the jacobian of func at point.
    """

    def _jacobian(point):
        """Compute the jacobian of func at each point of a batch."""
        return _apply_pointwise(
            functools.partial(_jacobian_at_point, func), point, point_ndim
        )

    return _jacobian


def _jacobian_and_hessian_at_point(func, point):
    """Compute the jacobian and the hessian of func at a single point.

    The hessian is the jacobian of the jacobian, whose value is the jacobian.
    """
    return _value_and_jacobian_at_point(
        functools.partial(_jacobian_at_point, func), point
    )


def hessian(func, point_ndim=1, func_out_ndim=None):
    """Return a function that returns the pointwise hessian of func.

    As for `jacobian`, the hessian of a batch of points is the batch of the
    hessians at each point. The hessian of a vector-valued func stacks the
    hessians of its components along the first axes.

    Parameters
    ----------
    func : callable
        Function whose hessian is computed.
    point_ndim : int
        Number of axes of a point. All preceding axes are batch axes.
    func_out_ndim : int
        Unused. Here for API consistency.

    Returns
    -------
    _ : callable
        Function taking point as input and returning
        the hessian of func at point.
    """

    def _hessian(point):
        """Compute the hessian of func at each point of a batch."""
        return _apply_pointwise(
            lambda one_point: _jacobian_and_hessian_at_point(func, one_point)[1],
            point,
            point_ndim,
        )

    return _hessian


def jacobian_and_hessian(func, point_ndim=1, func_out_ndim=None):
    """Return a function that returns func's pointwise jacobian and hessian.

    The jacobian is computed once and reused for the hessian.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian are computed.
    point_ndim : int
        Number of axes of a point. All preceding axes are batch axes.
    func_out_ndim : int
        Unused. Here for API consistency.

    Returns
    -------
    func_with_jacobian_and_hessian : callable
        Function that returns func's jacobian and
        func's hessian at its input point.
    """

    def _jacobian_and_hessian(point):
        """Compute the jacobian and hessian of func at each point of a batch."""
        return _apply_pointwise(
            functools.partial(_jacobian_and_hessian_at_point, func), point, point_ndim
        )

    return _jacobian_and_hessian


def value_and_grad(func, argnums=0, point_ndims=1):
    """Return a function that returns func's value and gradients' values.

    Suitable for use in scipy.optimize.

    Gradients are returned in the order specified by ``argnums`` without
    accumulating them in the input tensors' ``grad`` attributes.

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
        inputs = _broadcast_to_batch(inputs, point_ndims)
        for index in argnums_:
            if not inputs[index].requires_grad:
                inputs[index] = inputs[index].detach().requires_grad_(True)

        value = func(*inputs, **kwargs)

        grad_inputs = [inputs[index] for index in argnums_]
        grads = [None] * len(grad_inputs)
        if grad_inputs and value.requires_grad:
            grads = _torch.autograd.grad(
                value,
                grad_inputs,
                grad_outputs=_torch.ones_like(value),
                allow_unused=True,
            )

        grads = tuple(
            _torch.zeros_like(point) if grad is None else grad.detach()
            for point, grad in zip(grad_inputs, grads)
        )
        if isinstance(argnums, int):
            grads = grads[0]
        return value.detach(), grads

    return func_with_grad


def value_and_jacobian(func, point_ndim=1):
    """Return a function that returns func's value and pointwise jacobian.

    func is evaluated once for both.

    Parameters
    ----------
    func : callable
        Function whose jacobian is computed.
    point_ndim : int
        Number of axes of a point. All preceding axes are batch axes.

    Returns
    -------
    func_with_jacobian : callable
        Function that returns func's value and
        func's jacobian at its input point.
    """

    def _value_and_jacobian(point):
        """Compute the value and jacobian of func at each point of a batch."""
        return _apply_pointwise(
            functools.partial(_value_and_jacobian_at_point, func), point, point_ndim
        )

    return _value_and_jacobian


def _value_jacobian_and_hessian_at_point(func, point):
    """Compute the value, jacobian and hessian of func at a single point.

    func is evaluated once: its value is carried as auxiliary output of the
    jacobian, whose jacobian is the hessian.
    """

    def _jacobian_with_value(one_point):
        """Compute the jacobian of func, with the value as auxiliary output."""
        value, jac = _value_and_jacobian_at_point(func, one_point)
        return jac, value

    jac, hess, value = _value_and_jacobian_at_point(
        _jacobian_with_value, point, has_aux=True
    )
    return value, jac, hess


def value_jacobian_and_hessian(func, point_ndim=1, func_out_ndim=None):
    """Return a function that returns func's value, jacobian and hessian.

    func is evaluated once for the three of them.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian are computed.
    point_ndim : int
        Number of axes of a point. All preceding axes are batch axes.
    func_out_ndim : int
        Unused. Here for API consistency.

    Returns
    -------
    func_with_jacobian_and_hessian : callable
        Function that returns func's value, jacobian and
        hessian at its input point.
    """

    def _value_jacobian_and_hessian(point):
        """Compute the value, jacobian and hessian of func at each point of a batch."""
        return _apply_pointwise(
            functools.partial(_value_jacobian_and_hessian_at_point, func),
            point,
            point_ndim,
        )

    return _value_jacobian_and_hessian
