"""
Helper functions for unit tests.
"""

import numpy as np

# Left, from identity


def left_log_then_exp_from_identity(metric, point):
    aux = metric.left_log_from_identity(point=point)
    result = metric.left_exp_from_identity(tangent_vec=aux)
    return result


def left_exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.left_exp_from_identity(tangent_vec=tangent_vec)
    result = metric.left_log_from_identity(point=aux)
    return result

# From identity


def log_then_exp_from_identity(metric, point):
    aux = metric.log_from_identity(point=point)
    result = metric.exp_from_identity(tangent_vec=aux)
    return result


def exp_then_log_from_identity(metric, tangent_vec):
    aux = metric.exp_from_identity(tangent_vec=tangent_vec)
    result = metric.log_from_identity(point=aux)
    return result

# Standard


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


# -- Group


# From identity


def group_log_then_exp_from_identity(group, point):
    aux = group.group_log_from_identity(point=point)
    result = group.group_exp_from_identity(tangent_vec=aux)
    return result


def group_exp_then_log_from_identity(group, tangent_vec):
    aux = group.group_exp_from_identity(tangent_vec=tangent_vec)
    result = group.group_log_from_identity(point=aux)
    return result


# Standard


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


# Regularize a tangent_vector by getting its norm,
# at the base point, to be less than pi,
# following the regularization convention


def regularize_tangent_vec(group, tangent_vec, base_point):
    jacobian = group.jacobian_translation(
                                      point=base_point,
                                      left_or_right='left')
    tangent_vec_at_id = np.dot(np.linalg.inv(jacobian),
                               tangent_vec)
    tangent_vec_at_id = group.regularize(
                                        tangent_vec_at_id)
    regularized_tangent_vec = np.dot(jacobian, tangent_vec_at_id)
    return regularized_tangent_vec
