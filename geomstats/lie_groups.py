"""Computations on Lie groups."""

import numpy as np

from geomstats.base_manifolds import Manifold
from geomstats.base_manifolds import RiemannianMetric


class InvariantMetric(RiemannianMetric):
    """
    Base class for special Riemannian metrics that
    can be built on Lie groups:
    - left-invariant metrics
    - right-invariant metrics.

    Note: Assume that the points are parameterized by
    their Riemannian logarithm for the canonical left-invariant metric.
    """

    def __init__(self, lie_group, metric_matrix_at_identity,
                 left_or_right='left'):
        matrix_shape = metric_matrix_at_identity.shape
        assert matrix_shape == (lie_group.dimension, lie_group.dimension)
        assert left_or_right in ('left', 'right')

        self.lie_group = lie_group
        self.metric_matrix_at_identity = metric_matrix_at_identity
        self.left_or_right = left_or_right

    def metric_matrix(self, ref_point):
        """
        Compute the 6x6 matrix of the Riemmanian metric at point ref_point,
        by translating inner_product from the identity to ref_point.

        :param ref_point: 6D vector element of SE(3)
        :param inner_product: 6x6 matrix of inner product at the identity
        :param left_or_right: left/right translation of the inner product
        :returns metric_mat: 6x6 matrix of Riemannian metric at ref_point
        """
        ref_point = self.lie_group.regularize(ref_point)

        jacobian = self.lie_group.jacobian_translation(
                                  ref_point,
                                  left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)

        inv_jacobian_transposed = np.linalg.inv(jacobian.transpose())

        metric_mat = np.dot(inv_jacobian_transposed,
                            self.metric_matrix_at_identity)
        metric_mat = np.dot(metric_mat, inv_jacobian)

        return metric_mat

    def riemannian_left_exp_from_identity(self, tangent_vec):
        """
        Compute the *left* Riemannian exponential from the identity of the
        Lie group of tangent vector tangent_vec.

        The left Riemannian exponential has a special role since the
        left Riemannian exponential of the canonical metric parameterizes
        the points.
        """
        riem_exp = np.dot(self.metric_matrix_at_identity,
                          tangent_vec)

        riem_exp = self.lie_group.regularize(riem_exp)
        return riem_exp

    def riemannian_exp_from_identity(self, tangent_vec):
        """
        Compute the Riemannian exponential from the identity of the
        Lie group of tangent vector tangent_vec.
        """
        if self.left_or_right == 'left':
            riem_exp = self.riemannian_left_exp_from_identity(tangent_vec)

        else:
            opp_riem_left_exp = self.riemannian_left_exp_from_identity(
                                                             -tangent_vec)

            riem_exp = self.lie_group.inverse(opp_riem_left_exp)

        riem_exp = self.lie_group.regularize(riem_exp)
        return riem_exp

    def riemannian_exp(self, ref_point, tangent_vec):
        """
        Compute the Riemannian exponential at point ref_point
        of tangent vector tangent_vec.
        """
        ref_point = self.lie_group.regularize(ref_point)
        tangent_vec = self.lie_group.regularize(tangent_vec)

        jacobian = self.lie_group.jacobian_translation(
                                 ref_point,
                                 left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)

        tangent_vec_translated_to_id = np.dot(inv_jacobian, tangent_vec)

        exp_from_id = self.riemannian_exp_from_identity(
                               tangent_vec_translated_to_id)

        if self.left_or_right == 'left':
            exp = self.lie_group.compose(ref_point, exp_from_id)

        else:
            exp = self.lie_group.compose(exp_from_id, ref_point)

        exp = self.lie_group.regularize(exp)
        return exp

    def riemannian_left_log_from_identity(self, point):
        """
        Compute the *left* Riemannian logarithm from the identity of the
        Lie group of tangent vector tangent_vec.

        The left Riemannian logarithm has a special role since the
        left Riemannian logarithm of the canonical metric parameterizes
        the points.
        """
        inv_metric_matrix = np.linalg.inv(self.metric_matrix_at_identity)
        riem_log = np.dot(inv_metric_matrix, point)

        riem_log = self.lie_group.regularize(riem_log)
        return riem_log

    def riemannian_log_from_identity(self, point):
        """
        Compute the Riemannian logarithm of point at point ref_point
        of point for the invariant metric from the identity.
        """
        if self.left_or_right == 'left':
            riem_log = self.riemannian_left_log_from_identity(point)

        else:
            inv_point = self.lie_group.inverse(point)
            riem_left_log = self.riemannian_left_log_from_identity(inv_point)
            riem_log = - riem_left_log

        riem_log = self.lie_group.regularize(riem_log)
        return riem_log

    def riemannian_log(self, ref_point, point):
        """
        Compute the Riemannian logarithm of point at point ref_point
        of point for the invariant metric.
        """
        ref_point = self.lie_group.regularize(ref_point)
        point = self.lie_group.regularize(point)

        if self.left_or_right == 'left':
            point_near_id = self.lie_group.compose(
                                   self.lie_group.inverse(ref_point),
                                   point)

        else:
            point_near_id = self.lie_group.compose(
                                   point,
                                   self.lie_group.inverse(ref_point))

        log_from_id = self.riemannian_log_from_identity(point_near_id)

        jacobian = self.lie_group.jacobian_translation(
                                       ref_point,
                                       left_or_right=self.left_or_right)
        log = np.dot(jacobian, log_from_id)

        log = self.lie_group.regularize(log)
        return log


class LieGroup(Manifold):
    """ Base class for Lie groups."""

    def __init__(self, dimension, identity):
        Manifold.__init__(self, dimension)
        self.identity = identity
        self.riemannian_metrics = []

    def compose(self, point_a, point_b):
        """
        Composition of the Lie group.
        """
        raise NotImplementedError('The composition is not implemented.')

    def inverse(self, point):
        """
        Inverse law of the Lie group.
        """
        raise NotImplementedError('The inverse is not implemented.')

    def jacobian_translation(point, left_or_right='left'):
        """
        Compute the jacobian matrix of the differential
    of the left translation by the point.
        """
        raise NotImplementedError(
               'The jacobian of the Lie group translation is not implemented.')

    def group_exp(self, point, tangent_vec):
        """
        Compute the group exponential at point ref_point
        of tangent vector tangent_vec.
        """
        raise NotImplementedError(
                'The group exponential is not implemented.')

    def group_log(self, ref_point, point):
        """
        Compute the group logarithm at point ref_point
        of the point point.
        """
        raise NotImplementedError(
                'The group logarithm is not implemented.')

    def add_riemannian_metric(self, riemannian_metric):
        self.riemannian_metrics.append(riemannian_metric)
