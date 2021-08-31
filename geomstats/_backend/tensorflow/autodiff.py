import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfm = tfp.math


def detach(x):
    """Returns a new tensor detached from the current graph.

    Parameters
    ----------
    x : array-like
        Tensor to detach.
    """
    tf.stop_gradient(x)
    return x


def custom_gradient(*grad_funcs):
    """Decorate a function to define its custom gradient(s).

    Parameters
    ----------
    *grad_funcs : callables
        Custom gradient functions.
    """
    def wrapper(func):
        def func_with_grad(*args, **kwargs):
            def grad(upstream):
                if upstream.ndim < 1:
                    upstream = tf.expand_dims(upstream, axis=0)
                if upstream.ndim < 2:
                    upstream = tf.expand_dims(upstream, axis=1)
                grad_vals = []
                for grad_fun in grad_funcs:
                    grad_func_val = tf.convert_to_tensor(
                        grad_fun(*args, **kwargs))
                    grad_vals.append(
                        tf.squeeze(
                            tf.einsum(
                                "...,...->...", upstream, grad_func_val))
                    )
                return tuple(grad_vals)

            return func(*args, **kwargs), grad

        return tf.custom_gradient(func_with_grad)

    return wrapper


def value_and_grad(func, to_numpy=False):
    """Return a function that returns both value and gradient.

    Suitable for use in scipy.optimize

    Parameters
    ----------
    objective : callable
        Function to compute the gradient. It must be real-valued.
    to_numpy : bool
        Determines if the outputs value and grad will be cast
        to numpy arrays. Set to "True" when using scipy.optimize.
        Default: False.

    Returns
    -------
    objective_with_grad : callable
        Function that takes the argument of the objective function as input
        and returns both value and grad at the input.
    """
    def func_with_grad(*args):
        """Return the value of the function and its grad at the inputs."""
        if not isinstance(args, tuple):
            raise ValueError(
                "The inputs parameters are expected to form a tuple.")

        if isinstance(args[0], np.ndarray):
            args = (tf.Variable(one_arg) for one_arg in args)

        value, grad = tfm.value_and_gradient(func, *args)
        if to_numpy:
            return value.numpy(), grad.numpy()
        return value, grad

    return func_with_grad


def jacobian(func):
    """Return a function that returns the jacobian of a function func."""
    def jac(x):
        """Return the jacobian of func at x."""
        if isinstance(x, np.ndarray):
            x = tf.Variable(x)
        with tf.GradientTape() as g:
            g.watch(x)
            y = func(x)
        return g.jacobian(y, x)
    return jac
