import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfm = tfp.math

def detach(x):
    tf.stop_gradient(x)
    return x


def custom_gradient(*grad_funcs):
    """[Decorator to define a custom gradient to a function]

    Args:
        grad_func ([callable]): The custom gradient function
    """
    def wrapper(func):
        def func_returning_its_grad(*args, **kwargs):
            func_val = func(*args, **kwargs)

            if not isinstance(grad_funcs, tuple):
                def grad(upstream):
                    return upstream * grad_funcs(*args, **kwargs)
                return func_val, grad
        
            def grad_returning_tuple(upstream):
                grad_vals = (
                    upstream * grad_fun(*args, **kwargs)
                    for grad_fun in grad_funcs
                )
                return tuple(grad_vals)

            return func_val, grad_returning_tuple

        return tf.custom_gradient(func_returning_its_grad)

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
    def func_with_grad(*arg_x):
        """Return the value of the function and its grad at the inputs."""
        if not isinstance(arg_x, tuple):
            raise ValueError(
                "The inputs parameters are expected to form a tuple.")

        if isinstance(arg_x[0], np.ndarray):
            arg_x = (tf.Variable(one_arg_x) for one_arg_x in arg_x)

        value, grad = tfm.value_and_gradient(func, *arg_x)
        if to_numpy:
            return value.numpy(), grad.numpy()
        return value, grad

    return func_with_grad


def jacobian(f):
    """Return a function that returns the jacobian of a function f."""
    def jac(x):
        """Return the jacobian of f at x."""
        if isinstance(x, np.ndarray):
            x = tf.Variable(x)
        with tf.GradientTape() as g:
            g.watch(x)
            y = f(x)
        return g.jacobian(y, x)
    return jac
