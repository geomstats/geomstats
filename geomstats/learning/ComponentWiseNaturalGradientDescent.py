import torch

import geomstats.backend as gs


class ComponentWiseNaturalGradientDescent:
    """Implements Natural Gradient Descent using Van Sang et al. algorithm
    for convolutional neural networks
    Parameters
    ----------
    params : array-like
        Array of all the layers of the CNN, consisting of
        a vector of nodes, a vector of weights,
        and a bias scalar
    activations : array-like
        The linear outputs of the c
    layers_dict : OrderedDict
        Dictionary of all the layers of the CNN and their type
    lr : float
        The learning rate to update the weights
    gamma : float
        Float value added to ensure the Fisher matrix remains
        invertible for numerical reasons
    bias : Boolean
        Whether or not to include bias term in training
    References
    ----------
    .. [2210.05268] Tran Van Sang, Mhd Irvan,
    Rie Shigetomi Yamaguchi, Toshiyuki Nakata (2022)
    Component-Wise Natural Gradient Descent -- An Efficient Neural Network Optimization
    """

    def __init__(self, params, activations, layers_dict, lr=0.05, gamma=0.1, bias=True):
        self.defaults = dict(lr=lr, gamma=gamma, bias=bias)
        self.params = params
        self.layers_dict = layers_dict
        self.activations = activations

        self.dense_params = []
        self.conv_params = []
        self.conv_activations = []
        self.dense_activations = []
        self.dense_gradients = []
        self.conv_gradients = []

        for p in self.params:
            if p.dim() > 2:
                self.conv_params.append(p)
            else:
                self.dense_params.append(p)

    def zero_grad(self):
        """
        Zero out the gradients of all model parameters.

        This method sets the gradients of all model parameters to zero. It is typically
        called before backpropagation to clear the gradients computed in the previous
        iteration or batch. By zeroing out the gradients, we ensure that the gradients
        from different iterations or batches do not accumulate.

        Returns:
            None
        """
        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def step(self, gradients):
        """
        Update the parameters of the model using gradient descent.

        Parameters
        ----------
        gradients : array-like
            Array of the current gradients in the computation graph for
            the weights and biases, used to update the parameters

        Returns:
            None
        """
        for p in gradients:
            if p.dim() > 2:
                self.conv_gradients.append(p)
            else:
                self.dense_gradients.append(p)

        for a in self.activations:
            if a.dim() > 2:
                self.conv_activations.append(a)
            else:
                self.dense_activations.append(a)

        layer_grad = gradients[-2]
        layer_weight = self.params[-2]
        layer_bias = self.params[-1]
        W_prev = self.activations[-1]
        A_prev = torch.relu(W_prev)
        d_act_L = gs.where((gs.matmul(layer_weight, A_prev.squeeze().T) + layer_bias.unsqueeze(1)) > 0,
        gs.array(1.0), gs.array(0.0))
        D_a = gs.matmul(layer_grad, gs.matmul(d_act_L, A_prev.squeeze()).T)
        l2 = len(self.dense_params) // 2
        l3 = len(self.conv_params)
        for l1 in range(len(list(self.layers_dict.keys())), 1, -1):
            current_layer = list(self.layers_dict.keys())[l1-1]
            if current_layer[0:2] == 'fc':
                layer_grad = self.dense_gradients[2*l2-2]
                layer_weight = self.dense_params[2*l2-2]
                layer_bias = self.dense_params[2*l2-1]
                A_prev = self.dense_activations[l2-1]
                d_act_l = gs.where((gs.matmul(layer_weight, A_prev.T) + layer_bias.unsqueeze(1)) > 0,
gs.array(1.0), gs.array(0.0))
                D_s = gs.matmul(D_a, d_act_l)
                D_a_grad = gs.empty_like(D_s)
                D_a_grad = torch.autograd.grad(outputs=D_s, inputs=D_a, grad_outputs=D_a_grad)
                D_w = gs.matmul(A_prev.T, D_s.T)
                F = gs.matmul(D_w.T, D_w)
                D_w = D_w.unsqueeze(1)
                U = gs.matmul(D_w, gs.linalg.inv(F + self.defaults['gamma'])).squeeze().T
self.dense_params[2*l2-2].data -= self.defaults['lr'] * U
                l2 -= 2
            if current_layer[0:4] == 'conv':
                D_w = self.conv_gradients[l3-1]
                D_w = D_w.view(D_w.shape[3], -1)
                F = gs.matmul(D_w.T, D_w)
                U = gs.matmul(D_w, gs.linalg.inv(F + self.defaults['gamma']))
                self.conv_params[l3-1].data -= self.defaults['lr'] * U
                l3 -= 1
