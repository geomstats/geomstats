"""Wrapper around autograd functions to be consistent with backends."""

import funcsigs
from autograd import multigrad_dict
from autograd import numpy as np


from autograd import elementwise_grad as _elementwise_grad
from autograd import jacobian as _jacobian
from autograd import value_and_grad as _value_and_grad
from autograd.extend import defvjp, primitive


def detach(x):
    return x


def elementwise_grad(f):
    """Wrap autograd elementwise_grad function."""
    return _elementwise_grad(f)


def custom_gradient(*grad_func):
    """Decorate a function to define its custom gradient.

    Parameters
    ----------
    *grad_func : callables
        Custom gradient functions.
    """
    def decorator(function):

        wrapped_function = primitive(function)
        if len(grad_func) == 1:
            defvjp(
                wrapped_function,
                lambda ans, *args, **kwargs: lambda g: g * grad_func[0](*args, **kwargs))

        else:
            print(f"Number of grad functions: {len(grad_func)}")
            vjps = []
            for one_grad_func in grad_func:
                one_vjp = \
                    lambda ans, *args, **kwargs: lambda g: g * one_grad_func(
                        *args, **kwargs)
                vjps.append(one_vjp)
            vjps = tuple(vjps)

            defvjp(
                wrapped_function, 
                *vjps)

        return wrapped_function
    return decorator


def jacobian(f):
    """Wrap autograd jacobian function."""
    return _jacobian(f)


def value_and_grad(objective, to_numpy=False):
    """Wrap autograd value_and_grad function."""
    n_args = 2



    def aux_value_and_grad(x, y):
        def objective_rev(aux_y, aux_x):
            return objective(aux_x, aux_y)
        value, grad_x = _value_and_grad(objective)(x, y)
        _, grad_y = _value_and_grad(objective_rev)(y, x)

        def grad_y_rev(x, y):
            return grad_y(y, x)
        return value, (grad_x, grad_y)
    
    # def aux_value_and_grad(*args):
    #     value = objective(*args)



        # all_grads = []
        # for i in range(n_args):
        #     # if i == 0:
        #     #     objective_of_i = objective
        #     # else:
        #     def objective_of_i(*args):
        #         reorg_args =  args[1:i+1] + (args[0],) + args[i+1:]
        #         return objective(*reorg_args)
        #     new_args = (args[i],) + args[:i] + args[i+1:]
        #     value, grad_i = _value_and_grad(objective_of_i)(*new_args)
        #     all_grads.append(grad_i)

        #return value, tuple(all_grads)
    return aux_value_and_grad
    # def objective_one_concat(vector_arg):
    #     args = np.split(vector_arg, n_args)
    #     print(type(args))

    #     return objective(*args)

    # def aux_value_and_grad(*args):
    #     vector_arg = np.concatenate([args], axis=0)
    #     value, grad = _value_and_grad(objective_one_concat)(vector_arg)
    
    #     return value, tuple(np.split(grad, 2))
    #return aux_value_and_grad
    # if "_is_autograd_primitive" in objective.__dict__:
    #     multigradfunc_dict = multigrad_dict(objective.fun)
    # else:

    # multigradfunc_dict = multigrad_dict(objective)

    # def grads(*args, **kwargs):
    #     multigradvals_dict = multigradfunc_dict(*args, **kwargs)
    #     multigradvals = tuple(multigradvals_dict.values())
    #     multigradvals = multigradvals[0] if len(multigradvals) == 1 else multigradvals
    #     return multigradvals
    # return lambda *args, **kwargs: (objective(*args, **kwargs), grads(*args, **kwargs))
