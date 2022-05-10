"""Automatic differentiation in TensorFlow."""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfm = tfp.math


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
    tf.stop_gradient(x)
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
                    grads = tf.convert_to_tensor(grad_fun(*args, **kwargs))
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

        return tf.custom_gradient(func_with_grad)

    return decorator


def value_and_grad(func, to_numpy=False):
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

        if isinstance(args[0], np.ndarray):
            args = (tf.Variable(one_arg) for one_arg in args)

        value, grad = tfm.value_and_gradient(func, *args)
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

    def jac(x):
        """Return the jacobian of func at x.

        Parameters
        ----------
        x : array-like
            Input to function func or its jacobian.

        Returns
        -------
        _ : array-like
            Value of the jacobian of func at x.
        """
        if isinstance(x, np.ndarray):
            x = tf.Variable(x)
        with tf.GradientTape() as g:
            g.watch(x)
            y = func(x)
        return g.jacobian(y, x)

    return jac
