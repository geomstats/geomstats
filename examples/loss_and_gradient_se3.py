"""
Predict on SE3: losses.
"""
import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'  # NOQA
import geomstats.backend as gs
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
    if y_pred.ndim == 1:
        y_pred = gs.expand_dims(y_pred, axis=0)
    if y_true.ndim == 1:
        y_true = gs.expand_dims(y_true, axis=0)

    if representation == 'vector':
        grad = lie_group.grad(y_pred, y_true, SE3, metric)

    if representation == 'quaternion':

        y_pred_rot_vec = SO3.rotation_vector_from_quaternion(y_pred[:, :4])
        y_pred_pose = gs.hstack([y_pred_rot_vec, y_pred[:, 4:]])
        y_true_rot_vec = SO3.rotation_vector_from_quaternion(y_true[:, :4])
        y_true_pose = gs.hstack([y_true_rot_vec, y_true[:, 4:]])
        grad = lie_group.grad(y_pred_pose, y_true_pose, SE3, metric)

        differential = gs.zeros((1, 6, 7))

        upper_left_block = gs.zeros((1, 3, 4))
        lower_right_block = gs.zeros((1, 3, 3))
        quat_scalar = y_pred[:, :1]
        quat_vec = y_pred[:, 1:4]

        quat_vec_norm = gs.linalg.norm(quat_vec, axis=1)
        quat_sq_norm = quat_vec_norm ** 2 + quat_scalar ** 2
        # TODO(nina): check that this sq norm is 1?

        quat_arctan2 = gs.arctan2(quat_vec_norm, quat_scalar)
        differential_scalar = - 2 * quat_vec / (quat_sq_norm)
        differential_vec = (2 * (quat_scalar / quat_sq_norm
                                 - 2 * quat_arctan2 / quat_vec_norm)
                            * gs.outer(quat_vec, quat_vec) / quat_vec_norm ** 2
                            + 2 * quat_arctan2 / quat_vec_norm * gs.eye(3))

        upper_left_block[0, :, :1] = differential_scalar.transpose()
        upper_left_block[0, :, 1:] = differential_vec
        lower_right_block[0, :, :] = gs.eye(3)

        differential[0, :3, :4] = upper_left_block
        differential[0, 3:, 4:] = lower_right_block

        grad = gs.matmul(grad, differential)

    grad = gs.squeeze(grad, axis=0)
    return grad


def main():
    y_pred = gs.array([1., 1.5, -0.3, 5., 6., 7.])
    y_true = gs.array([0.1, 1.8, -0.1, 4., 5., 6.])

    loss_rot_vec = loss(y_pred, y_true)
    grad_rot_vec = grad(y_pred, y_true)
    print('The loss between the rotation vectors is: {}'.format(
        loss_rot_vec[0, 0]))
    print('The riemannian gradient is: {}'.format(
        grad_rot_vec[0]))

    angle = gs.pi / 6
    cos = gs.cos(angle / 2)
    sin = gs.sin(angle / 2)
    u = gs.array([1., 2., 3.])
    u = u / gs.linalg.norm(u)
    scalar = gs.array(cos)
    vec = sin * u
    translation = gs.array([5., 6., 7.])
    y_pred_quaternion = gs.hstack([scalar, vec, translation])

    angle = gs.pi / 7
    cos = gs.cos(angle / 2)
    sin = gs.sin(angle / 2)
    u = gs.array([1., 2., 3.])
    u = u / gs.linalg.norm(u)
    scalar = gs.array(cos)
    vec = sin * u
    translation = gs.array([4., 5., 6.])
    y_true_quaternion = gs.hstack([scalar, vec, translation])

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
