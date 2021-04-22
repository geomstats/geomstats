from functools import partial

import numpy as np
import tensorflow as tf


def custom_grad(grad_func):
    """[Decorator to define a custom gradient to a function]

    Args:
        grad_func ([callable]): The custom gradient function
    """

    def wrapper(func):
        def wrapped_func(*args, **kwargs):
            func_val = func(*args, **kwargs)
            return func_val, lambda grad_output: grad_func(*args, grad_output)

        return tf.custom_gradient(wrapped_func)

    return wrapper
    

def value_and_grad(objective):
    """'Returns a function that returns both value and gradient.

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
    '"""
    def objective_with_grad(velocity):
        if isinstance(velocity, np.ndarray):
            velocity = tf.Variable(velocity)
        with tf.GradientTape() as t:
            t.watch(velocity)
            loss = objective(velocity)
        return loss.numpy(), t.gradient(loss, velocity).numpy()
    return objective_with_grad
