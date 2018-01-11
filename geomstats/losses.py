"""
Predict on manifolds: losses.
"""

import geomstats.rigid_transformations as rigids


def rigids_riemannian_loss(
                    y_pred, y_true,
                    left_or_right='left',
                    inner_product=rigids.ALGEBRA_CANONICAL_INNER_PRODUCT):
    """
    Loss between two rigid transformations as
    the squared Riemannian distance.
    """
    riem_dist = rigids.riemannian_dist(y_pred, y_true,
                                       left_or_right=left_or_right,
                                       inner_product=inner_product)
    loss = riem_dist ** 2
    return loss


def rigids_riemannian_grad(
                    y_pred, y_true,
                    left_or_right='left',
                    inner_product=rigids.ALGEBRA_CANONICAL_INNER_PRODUCT):
    """
    Closed-form for the gradient of rigids_riemannian_loss.
    """
    tangent_vec = rigids.riemannian_log(y_true,
                                        ref_point=y_pred,
                                        left_or_right=left_or_right,
                                        inner_product=inner_product)
    grad = -2. * tangent_vec

    return grad
