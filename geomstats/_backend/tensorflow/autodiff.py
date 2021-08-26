import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tfp.math


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
            print(f"\n\nHere are the {len(args)} args:")
            print(args)
            print(f"\n\nHere are the {len(kwargs)} kkwargs:")
            print(kwargs)
            func_val = func(*args, **kwargs)

            if not isinstance(grad_funcs, tuple):
                def grad(upstream):
                    return upstream * grad_funcs(*args, **kwargs)
                return func_val, grad

            print(f"\n\nHere are the {len(grad_funcs)} grad_funcs:")
            print(grad_funcs)
        
            def grad_returning_tuple(upstream):
                grad_vals = []
                for grad_fun in grad_funcs:
                    grad_vals.append(upstream * grad_fun(*args, **kwargs))
                return tuple(grad_vals)

            return func_val, grad_returning_tuple

        return tf.custom_gradient(func_returning_its_grad)  # returns a modified function

    return wrapper  # returns the functional


    # def custom_gradient(*grad_funcs):
    # """[Decorator to define a custom gradient to a function]

    # Args:
    #     grad_func ([callable]): The custom gradient function
    # """

    # def wrapper(func):
    #     def wrapped_func(*args, **kwargs):
    #         func_val = func(*args, **kwargs)

    #         if not isinstance(grad_funcs, tuple):
    #             grad_vals = grad_funcs(*args, **kwargs)
    #             return func_val, lambda g: grad_vals * g
            
    #         grad_vals = []
    #         for grad_fun in grad_funcs:
    #             grad_vals.append(grad_fun(*args, **kwargs))
    #         grad_vals = tuple(grad_vals)

    #         return func_val, lambda g: tuple(k * g for k in grad_vals)

    #     return tf.custom_gradient(wrapped_func)

    # return wrapper


def value_and_grad(func):
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
        if isinstance(arg_x, np.ndarray):
             arg_x = tf.Variable(arg_x)
        return tfm.value_and_gradient(func, *arg_x)
    return func_with_grad


# def value_and_grad(func):
#     """Return a function that returns both value and gradient.

#     Suitable for use in scipy.optimize

#     Parameters
#     ----------
#     objective : callable
#         Function to compute the gradient. It must be real-valued.

#     Returns
#     -------
#     objective_with_grad : callable
#         Function that takes the argument of the objective function as input
#         and returns both value and grad at the input.
#     """
#     def func_with_grad(*arg_x):
#         if isinstance(arg_x, np.ndarray):
#             arg_x = tf.Variable(arg_x)
#         with tf.GradientTape() as t:
#             t.watch(*arg_x)
#             loss = func(*arg_x)
            
#         print("\n\n in autodiff")
#         print(loss)
#         print(arg_x)
#         loss_grad = t.gradient(loss, *arg_x)
#         return loss, loss_grad
#     return func_with_grad


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
