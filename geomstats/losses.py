"""
Predict on manifolds: losses.
"""
import numpy as np

from geomstats.special_euclidean_group import SpecialEuclideanGroup

SE3_GROUP = SpecialEuclideanGroup(n=3)


def lie_group_riemannian_loss(y_pred, y_true,
                              metric=SE3_GROUP.left_canonical_metric):
    """
    Loss function given by a riemannian metric on a Lie group,
    by default the left-invariant canonical metric.
    """
    loss = metric.squared_dist(y_pred, y_true)
    return loss


def lie_group_riemannian_numerical_grad_per_coord(y_pred, y_true,
                                                  delta=.001, coord=0):
    delta_vec = np.zeros(SE3_GROUP.dimension)
    delta_vec[coord] = delta

    y_pred_and_delta = y_pred + delta

    loss = lie_group_riemannian_loss(y_pred,
                                     y_true)
    loss_at_delta = lie_group_riemannian_loss(y_pred_and_delta,
                                              y_true)

    num_grad = (loss_at_delta - loss) / delta
    return num_grad


def lie_group_riemannian_grad(y_pred, y_true,
                              metric=SE3_GROUP.left_canonical_metric):
    """
    Closed-form for the gradient of lie_group_riemannian_loss.

    :return: tangent vector at point y_pred.
    """
    tangent_vec = metric.log(base_point=y_pred,
                             point=y_true)
    grad_point = - 2. * tangent_vec

    inner_prod_mat = metric.inner_product_matrix(base_point=y_pred)
    grad = np.dot(inner_prod_mat, grad_point)

    return grad
