"""
Predict on manifolds: losses.
"""
import numpy as np

from geomstats.special_euclidean_group import SpecialEuclideanGroup
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


SE3_GROUP = SpecialEuclideanGroup(n=3)
SO3_GROUP = SpecialOrthogonalGroup(n=3)


def pose_loss(y_pred, y_true, metric=SE3_GROUP.left_canonical_metric):
    """
    Loss function given by a riemannian metric on a Lie group,
    by default the left-invariant canonical metric.
    """
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=0)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=0)
    loss = metric.squared_dist(y_pred, y_true)
    return loss


def pose_grad(y_pred, y_true, metric=SE3_GROUP.left_canonical_metric):
    """
    Closed-form for the gradient of pose_loss.

    :return: tangent vector at point y_pred.
    """
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=0)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=0)
    tangent_vec = metric.log(base_point=y_pred,
                             point=y_true)
    grad_point = - 2. * tangent_vec

    inner_prod_mat = metric.inner_product_matrix(base_point=y_pred)

    grad = np.dot(grad_point, np.transpose(inner_prod_mat, axes=(0, 2, 1)))
    grad = np.squeeze(grad, axis=0)
    return grad


def quaternion_translation_loss(y_pred, y_true,
                                metric=SE3_GROUP.left_canonical_metric):
    """
    Loss function given by a riemannian metric on a Lie group,
    by default the left-invariant canonical metric.

    Here y_pred, y_true are of the form (quaternion, translation).
    """
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=0)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=0)
    y_pred_rot_vec = SO3_GROUP.rotation_vector_from_quaternion(y_pred[:, :4])
    y_pred = np.hstack([y_pred_rot_vec, y_pred[:, 4:]])
    y_true_rot_vec = SO3_GROUP.rotation_vector_from_quaternion(y_true[:, :4])
    y_true = np.hstack([y_true_rot_vec, y_true[:, 4:]])

    loss = pose_loss(y_pred, y_true, metric)
    return loss


def quaternion_translation_grad(y_pred, y_true,
                                metric=SE3_GROUP.left_canonical_metric):
    """
    Closed-form for the gradient of quaternion_translation_loss.

    Here y_pred, y_true are of the form (quaternion, translation).

    :return: tangent vector at point y_pred.
    """
    if y_pred.ndim == 1:
        y_pred = np.expand_dims(y_pred, axis=0)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=0)
    y_pred_rot_vec = SO3_GROUP.rotation_vector_from_quaternion(y_pred[:, :4])
    y_pred_pose = np.hstack([y_pred_rot_vec, y_pred[:, 4:]])
    y_true_rot_vec = SO3_GROUP.rotation_vector_from_quaternion(y_true[:, :4])
    y_true_pose = np.hstack([y_true_rot_vec, y_true[:, 4:]])
    grad_pose = pose_grad(y_pred_pose, y_true_pose, metric)

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

    grad = np.matmul(grad_pose, differential)
    grad = np.squeeze(grad, axis=0)
    return grad
