"""
Helper functions for unit tests.
"""

import numpy as np


def left_log_then_exp_from_identity(metric, point):
    aux = metric.left_log_from_identity(point=point)
    result = metric.left_exp_from_identity(tangent_vec=aux)
    return result


def left_exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.left_exp_from_identity(tangent_vec=tangent_vec)
    result = metric.left_log_from_identity(point=aux)
    return result


def log_then_exp_from_identity(metric, point):
    aux = metric.log_from_identity(point=point)
    result = metric.exp_from_identity(tangent_vec=aux)
    return result


def exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.exp_from_identity(tangent_vec=tangent_vec)
    result = metric.log_from_identity(point=aux)
    return result


def log_then_exp(metric, point, base_point):
    aux = metric.log(point=point,
                     base_point=base_point)
    result = metric.exp(tangent_vec=aux,
                        base_point=base_point)
    return result


def exp_then_log(metric, tangent_vec, base_point):
    aux = metric.exp(tangent_vec=tangent_vec,
                     base_point=base_point)
    result = metric.log(point=aux,
                        base_point=base_point)
    return result


def group_log_then_exp_from_identity(group, point):
    aux = group.group_log_from_identity(point=point)
    result = group.group_exp_from_identity(tangent_vec=aux)
    return result


def group_exp_then_log_from_identity(group, tangent_vec):
    aux = group.group_exp_from_identity(tangent_vec=tangent_vec)
    result = group.group_log_from_identity(point=aux)
    return result


def group_log_then_exp(group, point, base_point):
    aux = group.group_log(point=point,
                          base_point=base_point)
    result = group.group_exp(tangent_vec=aux,
                             base_point=base_point)
    return result


def group_exp_then_log(group, tangent_vec, base_point):
    aux = group.group_exp(tangent_vec=tangent_vec,
                          base_point=base_point)
    result = group.group_log(point=aux,
                             base_point=base_point)
    return result


def regularize_tangent_vec(group, tangent_vec, base_point):
    """
    Regularize a tangent_vector by getting its norm,
    at the base point, to be less than pi,
    following the regularization convention
    """
    base_point = group.regularize(base_point)
    if tangent_vec.ndim == 1:
        tangent_vec = np.expand_dims(tangent_vec, axis=0)
    assert tangent_vec.ndim == 2
    jacobian = group.jacobian_translation(
                                      point=base_point,
                                      left_or_right='left')
    inv_jacobian = np.linalg.inv(jacobian)
    tangent_vec_at_id = np.dot(tangent_vec,
                               np.transpose(inv_jacobian, axes=(0, 2, 1)))
    tangent_vec_at_id = np.squeeze(tangent_vec_at_id, axis=1)
    tangent_vec_at_id = group.regularize(tangent_vec_at_id)

    regularized_tangent_vec = np.dot(tangent_vec_at_id,
                                     np.transpose(jacobian, axes=(0, 2, 1)))
    regularized_tangent_vec = np.squeeze(regularized_tangent_vec, axis=1)
    return regularized_tangent_vec
