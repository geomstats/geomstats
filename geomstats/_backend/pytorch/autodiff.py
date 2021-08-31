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
                ctx.save_for_backward(*args)
                return function(*args)

            @staticmethod
            def backward(ctx, *grad_output):

                inputs = ctx.saved_tensors

                grads = tuple()

                for custom_grad, g in zip(args, grad_output):
                    grads = (*grads, custom_grad(*inputs) * g.clone())

                return grads

        def wrapper(*args):
            out = function_with_grad.apply(*args)
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
    def objective_with_grad(*arg_x):
        if len(arg_x) == 1:
            arg_x = arg_x[0]
            if isinstance(arg_x, np.ndarray):
                arg_x = torch.from_numpy(arg_x)
            arg_x = arg_x.clone().detach().requires_grad_(True)
            value = func(arg_x)
            if value.ndim > 0:
                value.backward(gradient=torch.ones_like(arg_x))
            else:
                value.backward()
            return value, torch.autograd.grad(func(arg_x), arg_x)[0]
        else:
            new_arg_x = []
            for one_arg in arg_x:
                if isinstance(one_arg, float):
                    one_arg = torch.from_numpy(np.array(one_arg))
                if isinstance(one_arg, np.ndarray):
                    one_arg = torch.from_numpy(one_arg)
                one_arg = one_arg.clone().detach().requires_grad_(True)
                new_arg_x.append(one_arg)
            arg_x = tuple(new_arg_x)

            value = func(*arg_x)
            if value.ndim > 0:
                value.backward(gradient=torch.ones_like(one_arg))
            else:
                value.backward()
            all_grads = []
            for one_arg in arg_x:
                all_grads.append(torch.autograd.grad(func(*arg_x), one_arg)[0])
            return value, tuple(all_grads)
    return objective_with_grad
