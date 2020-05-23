import numpy as np
import torch


def value_and_grad(objective):
    def objective_with_grad(velocity):
        """Create helpful objective func wrapper for autograd comp."""
        if isinstance(velocity, np.ndarray):
            velocity = torch.from_numpy(velocity)
        vel = velocity.clone().detach().requires_grad_(True)
        loss = objective(vel)
        loss.backward()
        return loss.detach().numpy(), vel.grad.detach().numpy()
    return objective_with_grad
