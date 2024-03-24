"""Natural gradient descent using Amari's anaytic solution.

References
----------
.. [AS1985] Amari S., Karakida R., Oizumi M. (2018)
    Fisher Information and Natural Gradient Learning of Random Deep Networks.
"""

import math

import numpy as np
import torch
from torch.optim import Optimizer

import geomstats.backend as gs


class NaturalGradientDescent(Optimizer):
    """Natural Gradient Descent.

    Uses Amari's analytic solution for the inverse Fisher information matrix.

    Parameters
    ----------
    params : array-like, shape=[3, dim]
        The current layer of the ResNet, consisting of a vector of nodes,
        a vector of weights, and a bias scalar
    lr : float
        The learning rate to update the weights
    wd : float
        Penalization on weights to prevent overfitting
    bias : Boolean
        Whether or not to include bias term in training

    References
    ----------
    .. [AS1985] Amari S., Karakida R., Oizumi M. (2018)
        Fisher Information and Natural Gradient Learning of Random Deep Networks.
    """

    def __init__(self, params, lr=0.05, wd=0.0, bias=True):
        defaults = dict(lr=lr, wd=wd, bias=bias)
        self.params = params

        super().__init__(params, defaults)

    def step(self):
        """Perform one step of Natural Gradient Descent."""
        w_list = []
        w_star_list = []
        w_tilde_list = []
        for group in self.param_groups:  # only 1 group for now
            for i, param in enumerate(group["params"]):
                if i % 2 == 0:  # only want to capture the weight parameters
                    param2 = param.detach().numpy()
                    w_list.append(param2)
                    # Append each bias value to the corresponding weight vector
                    bias_param = group["params"][i + 1]
                    param_concat_bias = torch.cat(
                        (param.data, bias_param.view(-1, 1)), dim=1
                    )
                    param2 = param_concat_bias.detach().numpy()
                    w_star_list.append(param2)

            for w_layer in w_list:
                for w in w_layer:
                    w2 = np.append(w, 0.0)
                    w_tilde_list.append(w2)

            for n_l, weight_layer in enumerate(w_list):
                for n_u, w in enumerate(weight_layer):
                    n = len(w)
                    x = torch.randn(
                        n
                    )  # approximate value of current node since we assume x ~ N(0, I)
                    w_star = w_star_list[n_l][n_u]
                    w_tilde = w_tilde_list[n_l][n_u]
                    e_star_0 = np.zeros(n + 1)
                    e_star_0[-1] = 1
                    A_00 = (1 / 2 * gs.sqrt(2)) * math.erf(
                        w_star[-1] / gs.sqrt(gs.dot(w, w))
                    )  # see Appendix II, eq. 107
                    A_0n = (1 / gs.sqrt(2)) * (
                        math.exp((-1 / 2) * (w_star[-1] / gs.sqrt(gs.dot(w, w))) ** 2)
                    )  # see Appendix II, eq. 109
                    A_nn = ((1 / 2 * gs.sqrt(2)) * A_00) - (
                        (w_star[-1] / gs.sqrt(gs.dot(w, w))) * A_0n
                    )  # see Appendix II, eq. 110
                    D = A_00 * A_nn - A_0n**2  # pp. 15, eq. 79
                    X = (1 / D) * A_00 - (1 / A_00)  # pp. 15, eq. 78
                    Y = (-1) * A_0n / D  # pp. 15, eq. 78
                    Z = A_nn / D - (1 / A_00)  # pp. 15, eq. 78
                    if param.grad is None:
                        continue
                    if i % 2 == 0:  # updating the gradients for the weights only
                        d_p = (
                            (-1)
                            * param.grad.data
                            * group["lr"]
                            * (
                                (1 / A_00) * x
                                + (
                                    X / gs.dot(w, w) * gs.dot(w, x)
                                    + (Y / gs.sqrt(gs.dot(w, w))) * w
                                )
                            )
                        )  # pp. 15, eq. 81
                    else:  # updating the gradients for the bias only
                        d_p = (
                            (-1)
                            * param.grad.data
                            * group["lr"]
                            * (
                                (1 / A_00)
                                + Z
                                + (Y * gs.dot(w, x) / gs.sqrt(gs.dot(w, w)))
                                * w_star[-1]
                            )
                        )  # pp. 15, eq. 82
                    param.data.add_(d_p)
