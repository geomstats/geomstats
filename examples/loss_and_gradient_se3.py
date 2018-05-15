"""
Predict on SE3: losses.
"""
import numpy as np

import geomstats.lie_group as lie_group
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


SE3 = SpecialEuclideanGroup(n=3)
SO3 = SpecialOrthogonalGroup(n=3)


def loss(y_pred, y_true,
         metric=SE3.left_canonical_metric,
         representation='vector'):
    """
    Loss function given by a riemannian metric on a Lie group,
    by default the left-invariant canonical metric.
    """
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=0)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=0)

    if representation == 'quaternion':
        y_pred_rot_vec = SO3.rotation_vector_from_quaternion(y_pred[:, :4])
        y_pred = np.hstack([y_pred_rot_vec, y_pred[:, 4:]])
        y_true_rot_vec = SO3.rotation_vector_from_quaternion(y_true[:, :4])
        y_true = np.hstack([y_true_rot_vec, y_true[:, 4:]])

    loss = lie_group.loss(y_pred, y_true, SE3, metric)
    return loss


def grad(y_pred, y_true,
         metric=SE3.left_canonical_metric,
         representation='vector'):
    """
    Closed-form for the gradient of pose_loss.

    :return: tangent vector at point y_pred.
    """
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=0)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=0)

    if representation == 'vector':
        grad = lie_group.grad(y_pred, y_true, SE3, metric)

    if representation == 'quaternion':

        y_pred_rot_vec = SO3.rotation_vector_from_quaternion(y_pred[:, :4])
        y_pred_pose = np.hstack([y_pred_rot_vec, y_pred[:, 4:]])
        y_true_rot_vec = SO3.rotation_vector_from_quaternion(y_true[:, :4])
        y_true_pose = np.hstack([y_true_rot_vec, y_true[:, 4:]])
        grad = lie_group.grad(y_pred_pose, y_true_pose, SE3, metric)

        differential = np.zeros((1, 6, 7))

        upper_left_block = np.zeros((1, 3, 4))
        lower_right_block = np.zeros((1, 3, 3))
        quat_scalar = y_pred[:, :1]
        quat_vec = y_pred[:, 1:4]

        quat_vec_norm = np.linalg.norm(quat_vec, axis=1)
        quat_sq_norm = quat_vec_norm ** 2 + quat_scalar ** 2
        # TODO(nina): check that this sq norm is 1?

        quat_arctan2 = np.arctan2(quat_vec_norm, quat_scalar)
        differential_scalar = - 2 * quat_vec / (quat_sq_norm)
        differential_vec = (2 * (quat_scalar / quat_sq_norm
                                 - 2 * quat_arctan2 / quat_vec_norm)
                            * np.outer(quat_vec, quat_vec) / quat_vec_norm ** 2
                            + 2 * quat_arctan2 / quat_vec_norm * np.eye(3))

        upper_left_block[0, :, :1] = differential_scalar.transpose()
        upper_left_block[0, :, 1:] = differential_vec
        lower_right_block[0, :, :] = np.eye(3)

        differential[0, :3, :4] = upper_left_block
        differential[0, 3:, 4:] = lower_right_block

        grad = np.matmul(grad, differential)

    grad = np.squeeze(grad, axis=0)
    return grad


def main():
    y_pred = np.array([1., 1.5, -0.3, 5., 6., 7.])
    y_true = np.array([0.1, 1.8, -0.1, 4., 5., 6.])

    loss_rot_vec = loss(y_pred, y_true)
    grad_rot_vec = grad(y_pred, y_true)
    print('The loss between the rotation vectors is: {}'.format(
        loss_rot_vec[0, 0]))
    print('The riemannian gradient is: {}'.format(
        grad_rot_vec[0]))

    angle = np.pi / 6
    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    u = np.array([1., 2., 3.])
    u = u / np.linalg.norm(u)
    scalar = np.array(cos)
    vec = sin * u
    translation = np.array([5., 6., 7.])
    y_pred_quaternion = np.hstack([scalar, vec, translation])

    angle = np.pi / 7
    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    u = np.array([1., 2., 3.])
    u = u / np.linalg.norm(u)
    scalar = np.array(cos)
    vec = sin * u
    translation = np.array([4., 5., 6.])
    y_true_quaternion = np.hstack([scalar, vec, translation])

    loss_quaternion = loss(y_pred_quaternion, y_true_quaternion,
                           representation='quaternion')
    grad_quaternion = grad(y_pred_quaternion, y_true_quaternion,
                           representation='quaternion')
    print('The loss between the quaternions is: {}'.format(
        loss_quaternion[0, 0]))
    print('The riemannian gradient is: {}'.format(
        grad_quaternion[0]))


if __name__ == "__main__":
    main()
