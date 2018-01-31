"""
Predict on manifolds: losses.
"""
import numpy as np

from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup
import tests.helper as helper


SE3_GROUP = SpecialEuclideanGroup(n=3)
SO3_GROUP = SpecialOrthogonalGroup(n=3)


def lie_group_riemannian_loss(y_pred, y_true,
                              metric=SE3_GROUP.left_canonical_metric):
    """
    Loss function given by a riemannian metric on a Lie group,
    by default the left-invariant canonical metric.
    """
    loss = metric.squared_dist(y_pred, y_true)
    return loss


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
    grad_rot = helper.regularize_tangent_vec(
                                           group=SO3_GROUP,
                                           tangent_vec=grad[:3],
                                           base_point=y_pred[:3])
    return grad
