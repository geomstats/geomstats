import numpy as np
import torch
from torch.autograd.functional import jacobian as torch_jac


def custom_gradient(*args):
    """[Decorator to define a custom gradient to a function (or multiple custom-gradient functions)]
    Args:
        *args : ([callables]): Custom gradient functions
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
                    grads = (*grads, custom_grad(*inputs)*g.clone())

                return grads

        def wrapper(*args):
                out = function_with_grad.apply(*args) 
                return out
            
        return wrapper
    return decorator


def value_and_grad(objective):
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
    def objective_with_grad(velocity):
        if isinstance(velocity, np.ndarray):
            velocity = torch.from_numpy(velocity)
        vel = velocity.clone().detach().requires_grad_(True)
        loss = objective(vel)
        if loss.ndim > 0:
            loss.backward(gradient=torch.ones_like(vel))
        else:
            loss.backward()
        return loss.detach().numpy(), vel.grad.detach().numpy()
    return objective_with_grad


def jacobian(f):
    """Return a function that returns the jacobian of a function."""
    return lambda x: torch_jac(f, x)
