"""Natural gradient descent using Amari's anaytic solution.

References
----------
.. [AS1985] Amari S., Karakida R., Oizumi M. (2018)
    Fisher Information and Natural Gradient Learning of Random Deep Networks.
"""

import math
import random

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
        """Performs one step of Natural Gradient Descent"""
        w_list = []  
        w_star_list = [] 
        for group in self.param_groups: # only 1 group for now
            for i, param in enumerate(group['params']):
                if i % 2 == 0: # only want to capture the weight parameters
                    param2 = param.detach().numpy()
                    w_list.append(param2)
                    # Append each bias value to the corresponding weight vector
                    bias_param = group["params"][i + 1]
                    param_concat_bias = torch.cat(
                        (param.data, bias_param.view(-1, 1)), dim=1
                    )
                    param2 = param_concat_bias.detach().numpy()
                    w_star_list.append(param2)

            for layer_index, weight_layer in enumerate(w_list):
                A_00_list = []
                D_list = []
                X_list = []
                Y_list = []
                Z_list = []
                for n_u, w in enumerate(weight_layer):
                    n = len(w)
                    x = torch.randn(n) # approximate value of current node since we assume x ~ N(0, I)
                    w_star = w_star_list[layer_index][n_u]
                    A_00 = (1/2*gs.sqrt(2)) * math.erf(w_star[-1] / gs.sqrt(gs.dot(w,w))) # see Appendix II, eq. 107
                    A_0n = (1/gs.sqrt(2)) * (math.exp((-1/2) * (w_star[-1] / gs.sqrt(gs.dot(w,w)))**2)) # see Appendix II, eq. 109
                    A_nn = ((1/2*gs.sqrt(2)) * A_00) -  ((w_star[-1] / gs.sqrt(gs.dot(w,w))) * A_0n) # see Appendix II, eq. 110
                    D = A_00 * A_nn - A_0n**2 # pp. 15, eq. 79
                    X = (1/D) * A_00 - (1 / A_00) # pp. 15, eq. 78
                    Y = (-1) * A_0n / D # pp. 15, eq. 78
                    Z = A_nn / D - (1 / A_00) # pp. 15, eq. 78
                    A_00_list.append(A_00)
                    D_list.append(D)
                    X_list.append(X)
                    Y_list.append(Y)
                    Z_list.append(Z)

                for i, param in enumerate(group['params'][layer_index: layer_index+2]):
                    for n_u, w in enumerate(weight_layer):
                        w_star = w_star_list[layer_index][n_u]
                        A_00 = A_00_list[n_u]
                        D = D_list[n_u]
                        X = X_list[n_u]
                        Y = Y_list[n_u]
                        Z = Z_list[n_u] 
                        alpha = group['lr']
                        if param.grad is None:
                            continue
                        if i % 2 == 0: # updating the gradients for the weights only
                          #  print('param grad', param.grad.data)
                            d_p = (-1) * param.grad.data * alpha * ((1/A_00) * x + \
                                (X / gs.dot(w,w) * gs.dot(w,x) + (Y / gs.sqrt(gs.dot(w,w))) * w)) # pp. 15, eq. 81
                            print(d_p)
                        else: # updating the gradients for the bias only
                            d_p = (-1) * param.grad.data * alpha * \
                                ((1/A_00) + Z + (Y * gs.dot(w, x) / gs.sqrt(gs.dot(w,w))) * 
                                 w_star[-1]) # pp. 15, eq. 82
                        with torch.no_grad():
                            print('x', x)
                            print('A00', A_00)
                            print('D', D)
                            print('X', X)
                            print('Y', Y)
                            print('Z', Z)
                            param.data.add_(d_p)
