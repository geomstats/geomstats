"""Automatic differentiation in TensorFlow."""

import numpy as _np
import tensorflow as _tf
import tensorflow_probability as _tfp

_tfm = _tfp.math


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
    _tf.stop_gradient(x)
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
        _ : callable
            Function func with gradients specified by grad_funcs.
        """

        def func_with_grad(*args, **kwargs):
            def grad(upstream):
                grad_vals = []
                for grad_fun in grad_funcs:
                    grads = _tf.convert_to_tensor(grad_fun(*args, **kwargs))
                    if isinstance(grads, float):
                        grad_val = upstream * grads
                    elif grads.ndim == 2:
                        grad_val = upstream[..., None] * grads
                    elif grads.ndim == 3:
                        grad_val = upstream[..., None, None] * grads
                    else:
                        grad_val = upstream * grads
                    grad_vals.append(grad_val)
                return tuple(grad_vals)

            return func(*args, **kwargs), grad

        return _tf.custom_gradient(func_with_grad)

    return decorator


def value_and_grad(func, to_numpy=False, create_graph=False):
    """Return a function that returns both value and gradient.

    Suitable for use in scipy.optimize with to_numpy=True.

    Parameters
    ----------
    func : callable
        Function whose value and gradient values
        will be computed. It must be real-valued.
    to_numpy : bool
        Determines if the outputs value and grad will be cast
        to numpy arrays. Set to "True" when using scipy.optimize.
        Default: False.
    create_graph : bool
        unused argument, set for compatibility with pytorch backend

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
        grad : any
            Values of func's gradients at input arguments args.
        """
        if not isinstance(args, tuple):
            raise ValueError("The inputs parameters are expected to form a tuple.")

        if isinstance(args[0], _np.ndarray):
            args = (_tf.Variable(one_arg) for one_arg in args)

        value, grad = _tfm.value_and_gradient(func, *args)
        if to_numpy:
            return value.numpy(), grad.numpy()
        return value, grad

    return func_with_grad


def jacobian(func):
    """Return a function that returns the jacobian of func.

    Parameters
    ----------
    func : callable
        Function whose Jacobian is computed.

    Returns
    -------
    jac : callable
        Function taking x as input and returning
        the jacobian of func at x.
    """

    def _jac(x):
        """Return the jacobian of func at x.

        Here, x is a single point of ndim 1.

        Parameters
        ----------
        x : array-like
            Input to function func or its jacobian.

        Returns
        -------
        _ : array-like
            Value of the jacobian of func at x.
        """
        if isinstance(x, _np.ndarray):
            x = _tf.Variable(x)
        with _tf.GradientTape() as g:
            g.watch(x)
            y = func(x)
        return g.jacobian(y, x)

    return _jac


def jacobian_vec(func):
    """Return a function that returns the jacobian of func.

    Parameters
    ----------
    func : callable
        Function whose Jacobian is computed.

    Returns
    -------
    jac : callable
        Function taking x as input and returning
        the jacobian of func at x.
    """

    def _jac(x):
        """Return the jacobian of func at x.

        Here, x is a single point of ndim 1.

        We note that the jacobian function of torch is not vectorized
        by default, thus we modify its behavior here.

        Default tensorflow behavior:

        If the jacobian for one point of shape (dim,) is of shape (out_dim, dim),
        then calling the jacobian on several points with shape (n_points, dim) will
        be of shape (out_dim, dim, n_points, dim).

        Modified behavior:

        Calling the jacobian on points gives a tensor of shape (n_points, out_dim, dim).

        Parameters
        ----------
        x : array-like
            Input to function func or its jacobian.

        Returns
        -------
        _ : array-like
            Value of the jacobian of func at x.
        """
        if isinstance(x, _np.ndarray):
            x = _tf.Variable(x)
        with _tf.GradientTape() as g:
            g.watch(x)
            y = func(x)
        return g.jacobian(y, x)

    def jac(x):
        """Return the jacobian of func at x.

        Here, x can be a batch of points.

        Parameters
        ----------
        x : array-like
            Input to function func or its jacobian.

        Returns
        -------
        _ : array-like
            Value of the jacobian of func at x.
        """
        if x.ndim == 1:
            return _jac(x)
        return _tf.vectorized_map(_jac, x)

    return jac


def hessian(func):
    """Return a function that returns the hessian of func.

    Parameters
    ----------
    func : callable
        Function whose Hessian is computed.

    Returns
    -------
    hess : callable
        Function taking x as input and returning
        the hessian of func at x.
    """

    def _hess(x):
        """Return the hessian of func at x.

        Parameters
        ----------
        x : array-like
            Input to function func or its hessian.

        Returns
        -------
        _ : array-like
            Value of the hessian of func at x.
        """
        # Note: this is a temporary implementation
        # that uses the jacobian of the gradient.
        # inspired from https://github.com/tensorflow/tensorflow/issues/29781
        # waiting for the hessian function to be implemented in GradientTape.
        if isinstance(x, _np.ndarray):
            x = _tf.Variable(x)

        with _tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = func(x)
            grads = g.gradient(y, [x])

        hessians = g.jacobian(grads[0], [x])
        return hessians[0]

    return _hess


def hessian_vec(func):
    """Return a function that returns the hessian of func.

    Parameters
    ----------
    func : callable
        Function whose Hessian is computed.

    Returns
    -------
    hess : callable
        Function taking x as input and returning
        the hessian of func at x.
    """

    def _hess(x):
        """Return the hessian of func at x.

        Parameters
        ----------
        x : array-like
            Input to function func or its hessian.

        Returns
        -------
        _ : array-like
            Value of the hessian of func at x.
        """
        # Note: this is a temporary implementation
        # that uses the jacobian of the gradient.
        # inspired from https://github.com/tensorflow/tensorflow/issues/29781
        # waiting for the hessian function to be implemented in GradientTape.
        if isinstance(x, _np.ndarray):
            x = _tf.Variable(x)

        with _tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = func(x)
            grads = g.gradient(y, [x])

        hessians = g.jacobian(grads[0], [x])
        return hessians[0]

    def hess(x):
        if x.ndim == 1:
            return _hess(x)
        return _tf.vectorized_map(_hess, x)

    return hess


def jacobian_and_hessian(func):
    """Return a function that returns the jacobian and hessian of func.

    Parameters
    ----------
    func : callable
        Function whose Jacobian and Hessian are computed.

    Returns
    -------
    jac_and_hess : callable
        Function taking x as input and returning
        the jacobian and hessian of func at x.
    """

    def jac_and_hess(x):
        """Return the jacobian and hessian of func at x.

        Parameters
        ----------
        x : array-like
            Input to function func or its jacobian and hessian.

        Returns
        -------
        _ : array-like
            Value of the jacobian and hessian of func at x.
        """
        # Note: this is a temporary implementation
        # that uses the jacobian of the gradient.
        # inspired from https://github.com/tensorflow/tensorflow/issues/29781
        # waiting for the hessian function to be implemented in GradientTape.
        if isinstance(x, _np.ndarray):
            x = _tf.Variable(x)

        with _tf.GradientTape(persistent=True) as g:
            g.watch(x)
            y = func(x)
            grads = g.gradient(y, [x])

        hessians = g.jacobian(grads[0], [x])
        return grads[0], hessians[0]

    return jac_and_hess
