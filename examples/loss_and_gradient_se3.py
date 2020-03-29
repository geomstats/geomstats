"""
Predict on SE3: losses.
"""

import logging

import geomstats.backend as gs
import geomstats.geometry.lie_group as lie_group
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


SE3 = SpecialEuclidean(n=3)
SO3 = SpecialOrthogonal(n=3)


def loss(y_pred, y_true,
         metric=SE3.left_canonical_metric,
         representation='vector'):
    """
    Loss function given by a riemannian metric on a Lie group,
    by default the left-invariant canonical metric.
    """
    if gs.ndim(y_pred) == 1:
        y_pred = gs.expand_dims(y_pred, axis=0)
    if gs.ndim(y_true) == 1:
        y_true = gs.expand_dims(y_true, axis=0)

    if representation == 'quaternion':
        y_pred_rot_vec = SO3.rotation_vector_from_quaternion(y_pred[:, :4])
        y_pred = gs.hstack([y_pred_rot_vec, y_pred[:, 4:]])
        y_true_rot_vec = SO3.rotation_vector_from_quaternion(y_true[:, :4])
        y_true = gs.hstack([y_true_rot_vec, y_true[:, 4:]])

    loss = lie_group.loss(y_pred, y_true, SE3, metric)
    return loss


def grad(y_pred, y_true,
         metric=SE3.left_canonical_metric,
         representation='vector'):
    """
    Closed-form for the gradient of pose_loss.

    :return: tangent vector at point y_pred.
    """
    if gs.ndim(y_pred) == 1:
        y_pred = gs.expand_dims(y_pred, axis=0)
    if gs.ndim(y_true) == 1:
        y_true = gs.expand_dims(y_true, axis=0)

    if representation == 'vector':
        grad = lie_group.grad(y_pred, y_true, SE3, metric)

    if representation == 'quaternion':

        y_pred_rot_vec = SO3.rotation_vector_from_quaternion(y_pred[:, :4])
        y_pred_pose = gs.hstack([y_pred_rot_vec, y_pred[:, 4:]])
        y_true_rot_vec = SO3.rotation_vector_from_quaternion(y_true[:, :4])
        y_true_pose = gs.hstack([y_true_rot_vec, y_true[:, 4:]])
        grad = lie_group.grad(y_pred_pose, y_true_pose, SE3, metric)

        quat_scalar = y_pred[:, :1]
        quat_vec = y_pred[:, 1:4]

        quat_vec_norm = gs.linalg.norm(quat_vec, axis=1)
        quat_sq_norm = quat_vec_norm ** 2 + quat_scalar ** 2

        quat_arctan2 = gs.arctan2(quat_vec_norm, quat_scalar)
        differential_scalar = - 2 * quat_vec / (quat_sq_norm)
        differential_vec = (2 * (quat_scalar / quat_sq_norm
                                 - 2 * quat_arctan2 / quat_vec_norm)
                            * (gs.einsum('ni,nj->nij', quat_vec, quat_vec)
                               / quat_vec_norm * quat_vec_norm)
                            + 2 * quat_arctan2 / quat_vec_norm * gs.eye(3))

        differential_scalar_t = gs.transpose(differential_scalar, axes=(1, 0))

        upper_left_block = gs.hstack(
            (differential_scalar_t, differential_vec[0]))
        upper_right_block = gs.zeros((3, 3))
        lower_right_block = gs.eye(3)
        lower_left_block = gs.zeros((3, 4))

        top = gs.hstack((upper_left_block, upper_right_block))
        bottom = gs.hstack((lower_left_block, lower_right_block))

        differential = gs.vstack((top, bottom))
        differential = gs.expand_dims(differential, axis=0)

        grad = gs.einsum('ni,nij->ni', grad, differential)

    grad = gs.squeeze(grad, axis=0)
    return grad


def main():
    y_pred = gs.array([1., 1.5, -0.3, 5., 6., 7.])
    y_true = gs.array([0.1, 1.8, -0.1, 4., 5., 6.])

    loss_rot_vec = loss(y_pred, y_true)
    grad_rot_vec = grad(y_pred, y_true)

    logging.info('The loss between the poses using rotation '
                 'vectors is: {}'.format(loss_rot_vec[0, 0]))
    logging.info('The riemannian gradient is: {}'.format(grad_rot_vec))

    angle = gs.array(gs.pi / 6)
    cos = gs.cos(angle / 2)
    sin = gs.sin(angle / 2)
    u = gs.array([1., 2., 3.])
    u = u / gs.linalg.norm(u)
    scalar = gs.array(cos)
    vec = sin * u
    translation = gs.array([5., 6., 7.])
    y_pred_quaternion = gs.concatenate([[scalar], vec, translation], axis=0)

    angle = gs.array(gs.pi / 7)
    cos = gs.cos(angle / 2)
    sin = gs.sin(angle / 2)
    u = gs.array([1., 2., 3.])
    u = u / gs.linalg.norm(u)
    scalar = gs.array(cos)
    vec = sin * u
    translation = gs.array([4., 5., 6.])
    y_true_quaternion = gs.concatenate([[scalar], vec, translation], axis=0)

    loss_quaternion = loss(y_pred_quaternion, y_true_quaternion,
                           representation='quaternion')
    grad_quaternion = grad(y_pred_quaternion, y_true_quaternion,
                           representation='quaternion')
    logging.info('The loss between the poses using quaternions is: {}'.format(
        loss_quaternion[0, 0]))
    logging.info('The riemannian gradient is: {}'.format(
        grad_quaternion))


if __name__ == "__main__":
    main()
