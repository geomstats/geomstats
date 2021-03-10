import numpy as np
import tensorflow as tf


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
