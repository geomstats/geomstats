"""Wrapper around autograd functions to be consistent with backends."""

import autograd as _autograd
import autograd.numpy as _np
from autograd import hessian as _hessian
from autograd import jacobian
from autograd import value_and_grad as _value_and_grad
from autograd.extend import defvjp as _defvjp
from autograd.extend import primitive as _primitive


def detach(x):
    """Return a new tensor detached from the current graph.

    This is a placeholder in order to have consistent backend APIs.

    Parameters
    ----------
    x : array-like
        Tensor to detach.

    Returns
    -------
    x : array-like
        Tensor.
    """
    return x


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
        wrapped_function = _primitive(func)

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
            _defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
            )
        elif len(grad_funcs) == 2:
            _defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: wrapped_grad_func(0, ans, *args, **kwargs),
                lambda ans, *args, **kwargs: wrapped_grad_func(1, ans, *args, **kwargs),
            )
        elif len(grad_funcs) == 3:
            _defvjp(
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


def value_and_grad(func, to_numpy=False):
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
    func_with_grad : callable
        Function that returns func's value and
        func's gradients' values at its inputs args.
    """

    def func_with_grad(*args):
        """Return func's value and func's gradients' values at args.

        Parameters
        ----------
        args : list
            Argument to function func and its gradients.

        Returns
        -------
        value : any
            Value of func at input arguments args.
        _ : tuple or any
            Values of func's gradients at input arguments args.
        """
        n_args = len(args)
        value = func(*args)

        all_grads = []
        for i in range(n_args):

            def func_of_ith(*args):
                reorg_args = args[1 : i + 1] + (args[0],) + args[i + 1 :]
                return func(*reorg_args)

            new_args = (args[i],) + args[:i] + args[i + 1 :]
            _, grad_i = _value_and_grad(func_of_ith)(*new_args)
            all_grads.append(grad_i)

        if n_args == 1:
            return value, all_grads[0]
        return value, tuple(all_grads)

    return func_with_grad


@_autograd.differential_operators.unary_to_nary
def _value_and_jacobian(fun, x):
    # same as autograd.jacobian, but also returning ans
    vjp, ans = _autograd.differential_operators._make_vjp(fun, x)
    ans_vspace = _autograd.differential_operators.vspace(ans)
    jacobian_shape = ans_vspace.shape + _autograd.differential_operators.vspace(x).shape
    grads = map(vjp, ans_vspace.standard_basis())
    return ans, _np.reshape(_np.stack(grads), jacobian_shape)


def jacobian_vec(func):
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
    func : callable
        Function whose jacobian values
        will be computed.

    Returns
    -------
    func_with_jacobian : callable
        Function that returns func's jacobian
        values at its inputs args.
    """

    def _jac(x):
        if x.ndim == 1:
            return jacobian(func)(x)
        return _np.stack([jacobian(func)(one_x) for one_x in x])

    return _jac


def hessian(func):
    """Wrap autograd hessian function.

    We note that the hessian function of autograd returns a tensor
    of shape (1, dim, dim) when one x is given as input.

    For consistency with the other backend, we convert this to a tensor
    of shape (dim, dim).

    Parameters
    ----------
    func : callable
        Function whose hessian values
        will be computed.

    Returns
    -------
    func_with_hessian : callable
        Function that returns func's hessian
        values at its inputs args.
    """

    def _hess(x):
        return _hessian(func)(x)[0]

    return _hess


def hessian_vec(func):
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

    Returns
    -------
    func_with_hessian : callable
        Function that returns func's hessian
        values at its inputs args.
    """

    def _hess(x):
        if x.ndim == 1:
            return _hessian(func)(x)[0]
        return _np.stack([_hessian(func)(one_x)[0] for one_x in x])

    return _hess


def jacobian_and_hessian(func):
    """Wrap autograd jacobian and hessian functions.

    Parameters
    ----------
    func : callable
        Function whose jacobian and hessian values
        will be computed.

    Returns
    -------
    func_with_jacobian_and_hessian : callable
        Function that returns func's jacobian and
        func's hessian values at its inputs args.
    """
    return _value_and_jacobian(jacobian(func))
