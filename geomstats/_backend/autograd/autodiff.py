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
                lambda ans, *args: lambda g: g * grad_func[0](*args))

        else:
            print(f"Number of grad functions: {len(grad_func)}")
            # vjps = []
            # for one_grad_func in grad_func:
            #     one_vjp = \
            #         lambda ans, *args: lambda g: g * one_grad_func(
            #             *args)
            #     vjps.append(one_vjp)
            # vjps = tuple(vjps)

            print(grad_func[0])
            print(grad_func[1])
            defvjp(
                wrapped_function, 
                lambda ans, *args: lambda g: g * grad_func[0](*args),
                lambda ans, *args: lambda g: g * grad_func[1](*args))

        return wrapped_function
    return decorator


def jacobian(f):
    """Wrap autograd jacobian function."""
    return _jacobian(f)


def value_and_grad(objective, to_numpy=False):
    """Wrap autograd value_and_grad function."""
    # return _value_and_grad(objective)
    func_sign = funcsigs.signature(objective)
    print(func_sign)
    print(type(func_sign))
    print("func_sign.parameters")
    print((func_sign.parameters))
    # if len(func_sign.parameters) == 1 or "args" in func_sign.parameters:
    #     print('indeed, one args!')
    #     # def to_return(*args):
    #     #     aux = _value_and_grad(objective)(*args)
    #     #     return aux[0], np.squeeze(aux[1])

    #     return _value_and_grad(objective)
    # else:
        # sig = funcsigs.signature(objective)  # {"x": Param, "y": Param}
        # print(sig.parameters)
        # bindings = sig.bind(arg_x, arg_y)
        # print(bindings.arguments)
        # print("bindings.arguments")
    if "_is_autograd_primitive" in objective.__dict__:
        objective = objective.fun
    multigradfunc_dict = multigrad_dict(objective)
    def multigrad_val(*args):
        values = []
        grad_vals = multigradfunc_dict(*args)
        for one_grad in grad_vals.values(): #multigradfunc_dict.values():
            print(one_grad.shape)
            print(type(one_grad))
            values.append(one_grad)
        result = tuple(values) if len(values) >1 else values[0]
        return result
    return lambda *args: (objective(*args), multigrad_val(*args))
