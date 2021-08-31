import numpy as np
import torch
from torch.autograd.functional import jacobian as torch_jac


def detach(x):
    return x.detach()


def custom_gradient(*args):
    """Decorate a function to define its custom gradient.

    Parameters
    ----------
    *grad_func : callables
        Custom gradient functions.
    """

    def decorator(function):

        class function_with_grad(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                print('whare are the args now')
                print(args)
                ctx.save_for_backward(*args)
                #ctx.z = z  # z is not a tensor
                return function(*args)

            @staticmethod
            def backward(ctx, grad_output):
                inputs = ctx.saved_tensors

                grads = []
                for custom_grad in args:
                    grads.append(grad_output * custom_grad(*inputs))
                grads = tuple(grads)

                if len(grads) == 1:
                    return grads[0]
                return grads

        def wrapper(*args, **kwargs):
            new_inputs = args + tuple(kwargs.values())
            print("\n\n\n\nnew inputs in wrapper")
            print(len(new_inputs))
            out = function_with_grad.apply(*new_inputs)
            return out

        return wrapper
    return decorator


def jacobian(f):
    """Return a function that returns the jacobian of a function."""
    return lambda x: torch_jac(f, x)


def value_and_grad(func, to_numpy=False):
    """'Return a function that returns both value and gradient.

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
    def objective_with_grad(*args, **kwargs):
        new_args = []
        for one_arg in args:
            if isinstance(one_arg, float):
                one_arg = torch.from_numpy(np.array(one_arg))
            if isinstance(one_arg, np.ndarray):
                one_arg = torch.from_numpy(one_arg)
            one_arg = one_arg.clone().requires_grad_(True)
            new_args.append(one_arg)
        args = tuple(new_args)

        value = func(*args, **kwargs)
        if value.ndim > 0:
            value.backward(gradient=torch.ones_like(one_arg), retain_graph=True)
        else:
            value.backward(retain_graph=True)

        all_grads = []
        for one_arg in args:
            all_grads.append(torch.autograd.grad(value, one_arg, retain_graph=True)[0])
        if len(args) == 1:
            return value, all_grads[0]
        return value, tuple(all_grads)
    return objective_with_grad
