"""Wrapper around autograd functions to be consistent with backends."""

import functools as _functools

import autograd as _autograd
import autograd.numpy as _np
from autograd import jacobian as _autograd_jacobian
from autograd.tracer import getval as _getval


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

    batch_shape = _np.broadcast_shapes(
        *(
            _np.shape(input_)[: _np.ndim(input_) - point_ndim]
            for input_, point_ndim in zip(inputs, point_ndims)
        )
    )
    return [
        _np.broadcast_to(
            input_, batch_shape + _np.shape(input_)[_np.ndim(input_) - point_ndim :]
        )
        for input_, point_ndim in zip(inputs, point_ndims)
    ]


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
        wrapped_function = _autograd.extend.primitive(func)
        _functools.update_wrapper(wrapped_function, func)

        def wrapped_grad_func(i, ans, *args, **kwargs):
            grads = grad_funcs[i](*args, **kwargs)

            # one value per point: g gets the point axes of grads
            return lambda g: (
                _np.reshape(g, _np.shape(g) + (1,) * (_np.ndim(grads) - _np.ndim(g)))
                * grads
            )

        _autograd.extend.defvjp(
            wrapped_function,
            *(
                None if grad_func is None else _functools.partial(wrapped_grad_func, i)
                for i, grad_func in enumerate(grad_funcs)
            ),
        )

        return wrapped_function

    return decorator


@_autograd.differential_operators.unary_to_nary
def _elementwise_value_and_grad(fun, x):
    # same as autograd.elementwise_grad, but also returning ans
    vjp, ans = _autograd.differential_operators._make_vjp(fun, x)
    if _autograd.differential_operators.vspace(ans).iscomplex:
        raise TypeError("Elementwise_grad only applies to real-output functions.")

    return ans, vjp(_autograd.differential_operators.vspace(ans).ones())


def value_and_grad(func, argnums=0, point_ndims=1):
    """Wrap autograd value_and_grad function.

    Suitable for use in scipy.optimize.

    Parameters
    ----------
    func : callable
        Function whose value and gradient values
        will be computed.
    argnums: int or tuple[int]
        Specifies arguments to compute gradients with respect to.
    point_ndims: int or tuple[int]
        Specifies arguments ndim.

    Returns
    -------
    value_and_grad : callable
        Function that returns func's value and
        func's gradients' values at its inputs args.
    """

    def _value_and_grad(*inputs, **kwargs):
        inputs = _broadcast_to_batch(inputs, point_ndims)
        return _elementwise_value_and_grad(func, argnum=argnums)(*inputs, **kwargs)

    return _value_and_grad


@_autograd.differential_operators.unary_to_nary
def _value_and_jacobian_op(fun, x):
    # same as autograd.jacobian, but also returning ans
    vjp, ans = _autograd.differential_operators._make_vjp(fun, x)
    ans_vspace = _autograd.differential_operators.vspace(ans)
    jacobian_shape = ans_vspace.shape + _autograd.differential_operators.vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return ans, _np.reshape(_np.stack(grads), jacobian_shape)


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
    value_and_jacobian_at_point = _value_and_jacobian_op(func)

    def _value_and_jacobian(point):
        """Compute the value and jacobian of func at each point of a batch."""
        return _apply_pointwise(value_and_jacobian_at_point, point, point_ndim)

    return _value_and_jacobian


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
    points = _np.reshape(point, (-1,) + point.shape[batch_ndim:])
    results = [func_at_point(one_point) for one_point in points]

    def _stack(arrays):
        """Stack arrays along the batch axes."""
        stacked = _np.stack(arrays)
        return _np.reshape(stacked, batch_shape + stacked.shape[1:])

    if isinstance(results[0], tuple):
        return tuple(_stack(list(arrays)) for arrays in zip(*results))

    return _stack(results)


def jacobian(func, point_ndim=1):
    """Return a function that returns the pointwise jacobian of func.

    Contrary to the jacobian of autograd, the jacobian of a batch of points
    is the batch of the jacobians at each point: if the jacobian at a point
    of shape (2,) has shape (3, 2), then the jacobian at 4 points of shape
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
    jacobian_at_point = _autograd_jacobian(func)

    def _jacobian(point):
        return _apply_pointwise(jacobian_at_point, point, point_ndim)

    return _jacobian


def hessian(func, point_ndim=1, func_out_ndim=None):
    """Return a function that returns the pointwise hessian of func.

    As for `jacobian`, the hessian of a batch of points is the batch of the
    hessians at each point.

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
    hessian_at_point = _autograd.hessian(func)

    def _hessian(point):
        return _apply_pointwise(hessian_at_point, point, point_ndim)

    return _hessian


def jacobian_and_hessian(func, point_ndim=1, func_out_ndim=None):
    """Return a function that returns func's pointwise jacobian and hessian.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian values
        will be computed.
    point_ndim : int
        Number of axes of a point. All preceding axes are batch axes.
    func_out_ndim : int
        Unused. Here for API consistency.

    Returns
    -------
    func_with_jacobian_and_hessian : callable
        Function that returns func's jacobian and
        func's hessian values at its inputs args.
    """
    return value_and_jacobian(
        jacobian(func, point_ndim=point_ndim), point_ndim=point_ndim
    )


def _value_jacobian_and_hessian_at_point(func, point):
    """Compute the value, jacobian and hessian of func at a single point.

    func is evaluated once: the hessian is the jacobian of the jacobian,
    whose computation records the value of func.
    """
    values = []

    def _jacobian(one_point):
        """Compute the jacobian of func, recording its value."""
        value, jac = _value_and_jacobian_op(func)(one_point)
        values.append(value)
        return jac

    jac, hess = _value_and_jacobian_op(_jacobian)(point)
    return _getval(values[0]), jac, hess


def value_jacobian_and_hessian(func, point_ndim=1, func_out_ndim=None):
    """Compute value, jacobian and hessian.

    func is only called once.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian values
        will be computed.
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
            _functools.partial(_value_jacobian_and_hessian_at_point, func),
            point,
            point_ndim,
        )

    return _value_jacobian_and_hessian
