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
                print("in forward")
                print("ctx")
                print(ctx)
                print("args")
                print(args)
                print(f"there are {len(args)} args in forward")
                ctx.save_for_backward(*args)
                print("ctx")
                print(ctx)
                print("\n")
                return function(*args)

            @staticmethod
            def backward(ctx, grad_output):
                print("ctx")
                print(ctx)
                print("grad_output")
                print(grad_output)
                #grad_output = (torch.tensor(1.), torch.tensor(1.))

                inputs = ctx.saved_tensors
                print("inputs in backward")
                print(inputs)

                grads = []
                print("args?")
                print(args)

                for custom_grad in args:
                    grads.append(grad_output * custom_grad(*inputs))
                grads = tuple(grads)


                print("\ngrads")
                print(type(grads))
                print(grads)
                if len(grads) == 1:
                    return grads[0]
                return grads

        def wrapper(*args):
            print("using apply...")
            print("args")
            print(args)
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
    def objective_with_grad(*args):
        new_args = []
        for one_arg in args:
            if isinstance(one_arg, float):
                one_arg = torch.from_numpy(np.array(one_arg))
            if isinstance(one_arg, np.ndarray):
                one_arg = torch.from_numpy(one_arg)
            one_arg = one_arg.clone().requires_grad_(True)
            new_args.append(one_arg)
        args = tuple(new_args)

        x, y = args
        value = func(x, y)
        # print()
        # print(type(value))
        # print(value)
        # if value.ndim > 0:
        #     # TODO(nina): this might not work with several args
        #     value.backward(gradient=torch.ones_like(one_arg))
        # else:
        #value.backward()

        all_grads = []
        all_grads.append(torch.autograd.grad(value, x, retain_graph=True)[0])
        all_grads.append(torch.autograd.grad(value, y)[0])
        #all_grads.append(torch.autograd.grad(value, y, allow_unused=True)[0])
        if len(args) == 1:
            return value, all_grads[0]
        return value, tuple(all_grads)
    return objective_with_grad
