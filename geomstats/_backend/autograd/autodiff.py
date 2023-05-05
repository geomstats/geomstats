"""Wrapper around autograd functions to be consistent with backends."""

import autograd as _autograd
import autograd.numpy as _np
from autograd import jacobian


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

        def wrapped_grad_func(i, ans, *args, **kwargs):
            grads = grad_funcs[i](*args, **kwargs)
            if isinstance(grads, float):
                return lambda g: g * grads
            if grads.ndim == 2:
                return lambda g: g[..., None] * grads
            if grads.ndim == 3:
                return lambda g: g[..., None, None] * grads
            return lambda g: g * grads

        if len(grad_funcs) == 1:
            _autograd.extend.defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
            )
        elif len(grad_funcs) == 2:
            _autograd.extend.defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
                lambda ans, *args, **kwargs: wrapped_grad_func(1, ans, *args, **kwargs),
            )
        elif len(grad_funcs) == 3:
            _autograd.extend.defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
                lambda ans, *args, **kwargs: wrapped_grad_func(1, ans, *args, **kwargs),
                lambda ans, *args, **kwargs: wrapped_grad_func(2, ans, *args, **kwargs),
            )
        else:
            raise NotImplementedError(
                "custom_gradient is not yet implemented " "for more than 3 gradients."
            )

        return wrapped_function

    return decorator


def _grad(func, argnums=0):
    def _wrapped_grad(*x, **kwargs):
        if not hasattr(x[0], "ndim") or x[0].ndim < 2:
            return _autograd.grad(func, argnum=argnums)(*x, **kwargs)

        return _autograd.elementwise_grad(func, argnum=argnums)(*x, **kwargs)

    return _wrapped_grad


@_autograd.differential_operators.unary_to_nary
def _elementwise_value_and_grad(fun, x):
    # same as autograd.elementwise_grad, but also returning ans
    vjp, ans = _autograd.differential_operators._make_vjp(fun, x)
    if _autograd.differential_operators.vspace(ans).iscomplex:
        raise TypeError("Elementwise_grad only applies to real-output functions.")

    return ans, vjp(_autograd.differential_operators.vspace(ans).ones())


def value_and_grad(func, argnums=0, to_numpy=False):
    """Wrap autograd value_and_grad function.

    Parameters
    ----------
    func : callable
        Function whose value and gradient values
        will be computed.
    to_numpy : bool
        Unused. Here for API consistency.

    Returns
    -------
    value_and_grad : callable
        Function that returns func's value and
        func's gradients' values at its inputs args.
    """

    def _value_and_grad(*x, **kwargs):
        if not hasattr(x[0], "ndim") or x[0].ndim < 2:
            return _autograd.value_and_grad(func, argnum=argnums)(*x, **kwargs)
        return _elementwise_value_and_grad(func, argnum=argnums)(*x, **kwargs)

    return _value_and_grad


@_autograd.differential_operators.unary_to_nary
def _value_and_jacobian_op(fun, x):
    # same as autograd.jacobian, but also returning ans
    vjp, ans = _autograd.differential_operators._make_vjp(fun, x)
    ans_vspace = _autograd.differential_operators.vspace(ans)
    jacobian_shape = ans_vspace.shape + _autograd.differential_operators.vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return ans, _np.reshape(_np.stack(grads), jacobian_shape)


def _value_and_jacobian(fun, point_ndim=1):
    def _value_and_jacobian_vec(x):
        if x.ndim == point_ndim:
            return _value_and_jacobian_op(fun)(x)

        ans = []
        jac = []
        for one_x in x:
            ans_, jac_ = _value_and_jacobian_op(fun)(one_x)
            ans.append(ans_)
            jac.append(jac_)

        return _np.stack(ans), _np.stack(jac)

    return _value_and_jacobian_vec


def jacobian_vec(fun, point_ndim=1):
    """Wrap autograd jacobian function.

    We note that the jacobian function of autograd is not vectorized
    by default, thus we modify its behavior here.

    Default autograd behavior:

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
    fun : callable
        Function whose jacobian values
        will be computed.

    Returns
    -------
    func_with_jacobian : callable
        Function that returns func's jacobian
        values at its inputs args.
    """

    def _jac(x):
        if x.ndim == point_ndim:
            return jacobian(fun)(x)
        return _np.stack([jacobian(fun)(one_x) for one_x in x])

    return _jac


def hessian(fun, func_out_ndim=None):
    """Wrap autograd hessian function.

    For consistency with the other backend, we convert this to a tensor
    of shape (dim, dim).

    Parameters
    ----------
    func : callable
        Function whose hessian values
        will be computed.
    func_out_ndim : int
        Unused. Here for API consistency.

    Returns
    -------
    func_with_hessian : callable
        Function that returns func's hessian
        values at its inputs args.
    """

    def _hess(x):
        return _autograd.hessian(fun)(x)

    return _hess


def hessian_vec(func, point_ndim=1, func_out_ndim=None):
    """Wrap autograd hessian function.

    We note that the hessian function of autograd is not vectorized
    by default, thus we modify its behavior here.

    We force the hessian to return a tensor of shape (n_points, dim, dim)
    when several points are given as inputs.

    Parameters
    ----------
    func : callable
        Function whose hessian values
        will be computed.
    func_out_ndim : int
        Unused. Here for API consistency.

    Returns
    -------
    func_with_hessian : callable
        Function that returns func's hessian
        values at its inputs args.
    """
    hessian_func = hessian(func)

    def _hess(x):
        if x.ndim == point_ndim:
            return hessian_func(x)
        return _np.stack([hessian_func(one_x) for one_x in x])

    return _hess


def jacobian_and_hessian(func, func_out_ndim=None):
    """Wrap autograd jacobian and hessian functions.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian values
        will be computed.
    func_out_ndim : int
        Unused. Here for API consistency.

    Returns
    -------
    func_with_jacobian_and_hessian : callable
        Function that returns func's jacobian and
        func's hessian values at its inputs args.
    """
    return _value_and_jacobian(jacobian_vec(func))


def value_jacobian_and_hessian(func, func_out_ndim=None):
    """Compute value, jacobian and hessian.

    func is only called once.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian values
        will be computed.
    func_out_ndim : int
        Unused. Here for API consistency.
    """
    cache = []

    def _cached_value_and_jacobian(fun, return_cached=False):
        def _jac(x):
            ans, jac = _value_and_jacobian(fun)(x)
            if not return_cached:
                cache.append(ans)
                return jac

            value = _np.stack(cache)._value if len(cache) > 1 else cache[0]._value
            cache.clear()

            return value, ans, jac

        return _jac

    return _cached_value_and_jacobian(
        _cached_value_and_jacobian(func), return_cached=True
    )
