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
#              arg_x = tf.Variable(arg_x)
#         if isinstance(arg_x, tuple):
#             if isinstance(arg_x[0], np.ndarray):
#                 arg_x = (tf.Variable(one_arg_x) for one_arg_x in arg_x)
#         return tfm.value_and_gradient(func, *arg_x)
#     return func_with_grad

# def value_and_grad(objective):
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
#         if isinstance(arg_x, tuple) and len(arg_x) == 1:
#             arg_x = tf.Variable(arg_x[0])
#         if not isinstance(arg_x, tuple):
#             with tf.GradientTape() as t:
#                 t.watch(arg_x)
#                 loss = objective(arg_x)
#             return loss.numpy(), t.gradient(loss, arg_x).numpy()
#         else:
#             return arg_x
#     return func_with_grad


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
            print("TYPE OF ARGS")
            print(type(arg_x))
            # Case with several args
            print("---- SEVERAL args here!")
            if isinstance(arg_x[0], np.ndarray):
                arg_x = (tf.Variable(one_arg_x) for one_arg_x in arg_x)
            with tf.GradientTape() as t:
                print("\n\n\n\n in autodiff:")
                print("The func is:")
                print(func)
                print("The arg_x is:")
                print(arg_x)
                print("The watched arg_x is:")
                print(arg_x)
                loss = func(*arg_x)

                print("The loss, i.e. the value of the func is:")
                print(loss)
            return loss, t.gradient(loss, *arg_x)

    return func_with_grad



# def value_and_grad(func, to_numpy=False):
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
#         # Case with one unique arg
#         if isinstance(arg_x, np.ndarray):
#             print("The arg_x is a np_arary")
#             arg_x = tf.cast(tf.Variable(arg_x), dtype=tf.float32)
#         if isinstance(arg_x, tuple) and len(arg_x) == 1:
#             arg_x = tf.cast(tf.Variable(arg_x[0]), dtype=tf.float32)

#         if not isinstance(arg_x, tuple):
#             print("---- Only one arg here!")
#             with tf.GradientTape() as t:
#                 print("\n\n\n\n in autodiff:")
#                 print("The func is:")
#                 print(func)
#                 print("The arg_x is:")
#                 print(arg_x)
#                 t.watch(arg_x)
#                 loss = func(arg_x)

#                 print("The loss, i.e. the value of the func is:")
#                 print(loss)

#                 #loss_grad = t.gradient(loss, arg_x) ## bug here
#                 #print("The shape of the loss grad is:")
#                 #print(tf.expand_dims(loss_grad, 0).shape)
#                 #loss_grad = tf.expand_dims(loss_grad, 0).transpose()
#             return  loss.numpy(), t.gradient(loss, arg_x).numpy()
#         # Case with several args
#         print("---- SEVERAL args here!")
#         if isinstance(arg_x[0], np.ndarray):
#             arg_x = (tf.Variable(one_arg_x) for one_arg_x in arg_x)
#         with tf.GradientTape() as t:
#             print("\n\n\n\n in autodiff:")
#             print("The func is:")
#             print(func)
#             print("The arg_x is:")
#             print(arg_x)
#             t.watch(*arg_x)
#             print("The watched arg_x is:")
#             print(arg_x)
#             loss = func(*arg_x)

#             print("The loss, i.e. the value of the func is:")
#             print(loss)

#             loss_grad = t.gradient(loss, *arg_x) ## bug here
#         return loss.numpy(), loss_grad.numpy()

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
