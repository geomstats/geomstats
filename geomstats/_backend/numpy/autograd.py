import numpy as np # NOQA


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
        # TODO
        vel = velocity.clone().requires_grad_(True)
        loss = objective(vel)
        loss.backward()
        return loss, vel.grad
    return objective_with_grad
