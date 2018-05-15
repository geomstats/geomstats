"""
Predict on manifolds: losses.
"""
import numpy as np

import geomstats.lie_group as lie_group
from geomstats.special_orthogonal_group import SpecialOrthogonalGroup


SO3 = SpecialOrthogonalGroup(n=3)


def loss(y_pred, y_true,
         metric=SO3.bi_invariant_metric,
         representation='vector'):

    if representation == 'quaternion':
        y_pred = SO3.rotation_vector_from_quaternion(y_pred)
        y_true = SO3.rotation_vector_from_quaternion(y_true)

    loss = lie_group.loss(y_pred, y_true, metric)
    return loss


def grad(y_pred, y_true,
         metric=SO3.bi_invariant_metric,
         representation='vector'):

    if representation == 'quaternion':
        y_pred = SO3.rotation_vector_from_quaternion(y_pred)
        y_true = SO3.rotation_vector_from_quaternion(y_true)

    grad = lie_group.grad(y_pred, y_true, metric)

    if representation == 'quaternion':
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

def main():
    phantom_y_pred = np.array([1., 1.5, -0.3])
    phantom_y_true = np.array([0.1, 1.8, -0.1])



if __name__ == "__main__":
    main()
