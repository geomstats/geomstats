"""Predict on SO(3): losses."""

import logging

import geomstats.backend as gs
import geomstats.geometry.lie_group as lie_group
from geomstats.geometry.special_orthogonal import SpecialOrthogonal


SO3 = SpecialOrthogonal(n=3, point_type='vector')


def loss(y_pred, y_true,
         metric=SO3.bi_invariant_metric,
         representation='vector'):
    """Loss function given by a Riemannian metric on a Lie group.

    Parameters
    ----------
    y_pred : array-like
        Prediction on SO(3).
    y_true : array-like
        Ground-truth on SO(3).
    metric : RiemannianMetric
        Metric used to compute the loss and gradient.
    representation : str, {'vector', 'matrix'}
        Representation chosen for points in SO(3).

    Returns
    -------
    lie_loss : array-like
        Loss using the Riemannian metric.
    """
    if representation == 'quaternion':
        y_pred = SO3.rotation_vector_from_quaternion(y_pred)
        y_true = SO3.rotation_vector_from_quaternion(y_true)

    lie_loss = lie_group.loss(y_pred, y_true, SO3, metric)
    return lie_loss


def grad(y_pred, y_true,
         metric=SO3.bi_invariant_metric,
         representation='vector'):
    """Closed-form for the gradient of pose_loss.

    Parameters
    ----------
    y_pred : array-like
        Prediction on SO(3).
    y_true : array-like
        Ground-truth on SO(3).
    metric : RiemannianMetric
        Metric used to compute the loss and gradient.
    representation : str, {'vector', 'matrix'}
        Representation chosen for points in SE(3).

    Returns
    -------
    lie_grad : array-like
        Tangent vector at point y_pred.
    """
    y_pred = gs.expand_dims(y_pred, axis=0)
    y_true = gs.expand_dims(y_true, axis=0)

    if representation == 'vector':
        lie_grad = lie_group.grad(y_pred, y_true, SO3, metric)

    if representation == 'quaternion':
        quat_scalar = y_pred[:, :1]
        quat_vec = y_pred[:, 1:]

        quat_vec_norm = gs.linalg.norm(quat_vec, axis=1)
        quat_sq_norm = quat_vec_norm ** 2 + quat_scalar ** 2

        quat_arctan2 = gs.arctan2(quat_vec_norm, quat_scalar)
        differential_scalar = - 2 * quat_vec / (quat_sq_norm)
        differential_scalar = gs.to_ndarray(differential_scalar, to_ndim=2)
        differential_scalar = gs.transpose(differential_scalar)

        differential_vec = (2 * (quat_scalar / quat_sq_norm
                                 - 2 * quat_arctan2 / quat_vec_norm)
                            * (gs.einsum('ni,nj->nij', quat_vec, quat_vec)
                               / quat_vec_norm ** 2)
                            + 2 * quat_arctan2 / quat_vec_norm * gs.eye(3))
        differential_vec = gs.squeeze(differential_vec)

        differential = gs.concatenate(
            [differential_scalar, differential_vec],
            axis=1)

        y_pred = SO3.rotation_vector_from_quaternion(y_pred)
        y_true = SO3.rotation_vector_from_quaternion(y_true)

        lie_grad = lie_group.grad(y_pred, y_true, SO3, metric)

        lie_grad = gs.matmul(lie_grad, differential)

    lie_grad = gs.squeeze(lie_grad, axis=0)
    return lie_grad


def main():
    """Print loss and gradient on SO(3)."""
    y_pred = gs.array([1., 1.5, -0.3])
    y_true = gs.array([0.1, 1.8, -0.1])

    loss_rot_vec = loss(y_pred, y_true)
    grad_rot_vec = grad(y_pred, y_true)

    logging.info('The loss between the rotation vectors is: {}'.format(
        loss_rot_vec))
    logging.info('The Riemannian gradient is: {}'.format(
        grad_rot_vec))

    angle = gs.array(gs.pi / 6)
    cos = gs.cos(angle / 2)
    sin = gs.sin(angle / 2)
    u = gs.array([1., 2., 3.])
    u = u / gs.linalg.norm(u)
    scalar = gs.to_ndarray(cos, to_ndim=1)
    vec = sin * u
    y_pred_quaternion = gs.concatenate([scalar, vec], axis=0)

    angle = gs.array(gs.pi / 7)
    cos = gs.cos(angle / 2)
    sin = gs.sin(angle / 2)
    u = gs.array([1., 2., 3.])
    u = u / gs.linalg.norm(u)
    scalar = gs.to_ndarray(cos, to_ndim=1)
    vec = sin * u
    y_true_quaternion = gs.concatenate([scalar, vec], axis=0)

    loss_quaternion = loss(y_pred_quaternion, y_true_quaternion,
                           representation='quaternion')
    grad_quaternion = grad(y_pred_quaternion, y_true_quaternion,
                           representation='quaternion')

    logging.info('The loss between the quaternions is: {}'.format(
        loss_quaternion))
    logging.info('The Riemannian gradient is: {}'.format(
        grad_quaternion))


if __name__ == '__main__':
    main()
