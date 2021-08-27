import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tfp.math
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def detach(x):
    tf.stop_gradient(x)
    return x


def custom_gradient(*grad_funcs):
    """[Decorator to define a custom gradient to a function]

    Args:
        grad_func ([callable]): The custom gradient function
    """
    def wrapper(func):  # functional
        def func_returning_its_grad(*args, **kwargs):
            func_val = func(*args, **kwargs)

            if not isinstance(grad_funcs, tuple):
                def grad(upstream):
                    return upstream * grad_funcs(*args, **kwargs)
                return func_val, grad
        
            def grad_returning_tuple(upstream):
                grad_vals = []
                for grad_fun in grad_funcs:
                    grad_vals.append(upstream * grad_fun(*args, **kwargs))
                return tuple(grad_vals)

            return func_val, grad_returning_tuple

        return tf.custom_gradient(func_returning_its_grad)  # returns a modified function

    return wrapper  # returns the functional

def value_and_grad(func, to_numpy=False):
    """Return a function that returns both value and gradient.

    Suitable for use in scipy.optimize

    Parameters
    ----------
    objective : callable
        Function to compute the gradient. It must be real-valued.

    Returns
    -------
    objective_with_grad : callable
        Function that takes the argument of the objective function as input
        and returns both value and grad at the input.
    """
    def func_with_grad(*arg_x):
        # Case with one unique arg
        if isinstance(arg_x, np.ndarray):
            arg_x = tf.Variable(arg_x)
        if isinstance(arg_x, tuple) and len(arg_x) == 1:
            arg_x = tf.Variable(arg_x[0])

        if not isinstance(arg_x, tuple):
            with tf.GradientTape() as t:
                t.watch(arg_x)
                loss = func(arg_x)
            if to_numpy:
                return loss.numpy(), t.gradient(loss, arg_x).numpy()
            else:
                return loss, t.gradient(loss, arg_x)
        else:
            assert isinstance(arg_x, tuple)

            # Case with several args
            print(f"---- SEVERAL args here! {len(arg_x)}")
            if isinstance(arg_x[0], np.ndarray):
                arg_x = (tf.Variable(one_arg_x) for one_arg_x in arg_x)
            with tf.GradientTape() as t:
                for one_arg_x in arg_x:
                    t.watch(one_arg_x)
                loss = func(*arg_x)
            return loss, t.gradient(loss, *arg_x)

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
