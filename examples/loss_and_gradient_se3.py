"""Predict on SE3: losses."""

import logging

import geomstats.backend as gs
import geomstats.geometry.lie_group as lie_group
from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


SE3 = SpecialEuclidean(n=3, point_type='vector')
SO3 = SpecialOrthogonal(n=3, point_type='vector')


def loss(y_pred, y_true,
         metric=SE3.left_canonical_metric,
         representation='vector'):
    """Loss function given by a Riemannian metric on a Lie group.

    Parameters
    ----------
    y_pred : array-like
        Prediction on SE(3).
    y_true : array-like
        Ground-truth on SE(3).
    metric : RiemannianMetric
        Metric used to compute the loss and gradient.
    representation : str, {'vector', 'matrix'}
        Representation chosen for points in SE(3).

    Returns
    -------
    lie_loss : array-like
        Loss using the Riemannian metric.
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

    lie_loss = lie_group.loss(y_pred, y_true, SE3, metric)
    if gs.ndim(lie_loss) == 2:
        lie_loss = gs.squeeze(lie_loss, axis=1)
    if gs.ndim(lie_loss) == 1 and gs.shape(lie_loss)[0] == 1:
        lie_loss = gs.squeeze(lie_loss, axis=0)

    return lie_loss


def grad(y_pred, y_true,
         metric=SE3.left_canonical_metric,
         representation='vector'):
    """Closed-form for the gradient of pose_loss.

    Parameters
    ----------
    y_pred : array-like
        Prediction on SE(3).
    y_true : array-like
        Ground-truth on SE(3).
    metric : RiemannianMetric
        Metric used to compute the loss and gradient.
    representation : str, {'vector', 'matrix'}
        Representation chosen for points in SE(3).

    Returns
    -------
    lie_grad : array-like
        Tangent vector at point y_pred.
    """
    if gs.ndim(y_pred) == 1:
        y_pred = gs.expand_dims(y_pred, axis=0)
    if gs.ndim(y_true) == 1:
        y_true = gs.expand_dims(y_true, axis=0)

    if representation == 'vector':
        lie_grad = lie_group.grad(y_pred, y_true, SE3, metric)

    if representation == 'quaternion':

        y_pred_rot_vec = SO3.rotation_vector_from_quaternion(y_pred[:, :4])
        y_pred_pose = gs.hstack([y_pred_rot_vec, y_pred[:, 4:]])
        y_true_rot_vec = SO3.rotation_vector_from_quaternion(y_true[:, :4])
        y_true_pose = gs.hstack([y_true_rot_vec, y_true[:, 4:]])
        lie_grad = lie_group.grad(y_pred_pose, y_true_pose, SE3, metric)

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

        lie_grad = gs.einsum('ni,nij->ni', lie_grad, differential)

    lie_grad = gs.squeeze(lie_grad, axis=0)
    return lie_grad


def main():
    """Print loss and gradient on SE(3)."""
    y_pred = gs.array([1., 1.5, -0.3, 5., 6., 7.])
    y_true = gs.array([0.1, 1.8, -0.1, 4., 5., 6.])

    loss_rot_vec = loss(y_pred, y_true)
    grad_rot_vec = grad(y_pred, y_true)

    logging.info('The loss between the poses using rotation '
                 'vectors is: {}'.format(loss_rot_vec))
    logging.info('The Riemannian gradient is: {}'.format(grad_rot_vec))

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
        loss_quaternion))
    logging.info('The Riemannian gradient is: {}'.format(
        grad_quaternion))


if __name__ == '__main__':
    main()
