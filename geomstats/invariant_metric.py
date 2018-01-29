"""
Base class for special Riemannian metrics that
can be built on Lie groups:
- left-invariant metrics
- right-invariant metrics.

Note: Assume that the points are parameterized by
their Riemannian logarithm for the canonical left-invariant metric.
"""

import numpy as np

from geomstats.riemannian_metric import RiemannianMetric


class InvariantMetric(RiemannianMetric):
    """
    Base class for left- or right- invariant metrics
    that can be defined on Lie groups.
    """

    def __init__(self, lie_group, inner_product_mat_at_identity,
                 left_or_right='left'):
        matrix_shape = inner_product_mat_at_identity.shape
        assert matrix_shape == (lie_group.dimension, lie_group.dimension)
        assert left_or_right in ('left', 'right')

        eigenvalues = np.linalg.eigvalsh(inner_product_mat_at_identity)
        n_pos_eigval = np.sum(eigenvalues > 0)
        n_neg_eigval = np.sum(eigenvalues < 0)
        n_null_eigval = np.sum(eigenvalues == 0)
        assert (n_pos_eigval + n_neg_eigval
                + n_null_eigval) == lie_group.dimension

        self.lie_group = lie_group
        self.inner_product_mat_at_identity = inner_product_mat_at_identity
        self.left_or_right = left_or_right
        self.signature = (n_pos_eigval, n_null_eigval, n_neg_eigval)

    def inner_product_matrix(self, base_point):
        """
        Compute the matrix of the Riemmanian metric at point base_point,
        by translating inner_product from the identity to base_point.
        """
        base_point = self.lie_group.regularize(base_point)

        jacobian = self.lie_group.jacobian_translation(
                                  base_point,
                                  left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)

        inv_jacobian_transposed = np.linalg.inv(jacobian.transpose())

        metric_mat = np.dot(inv_jacobian_transposed,
                            self.inner_product_mat_at_identity)
        metric_mat = np.dot(metric_mat, inv_jacobian)

        return metric_mat

    def left_exp_from_identity(self, tangent_vec):
        """
        Compute the *left* Riemannian exponential from the identity of the
        Lie group of tangent vector tangent_vec.

        The left Riemannian exponential has a special role since the
        left Riemannian exponential of the canonical metric parameterizes
        the points.

        Note: In the case where the method is called by a right-invariant
        metric, it used the left-invariant metric associated to the same
        inner-product at the identity.
        """
        exp = np.dot(self.inner_product_mat_at_identity, tangent_vec)

        exp = self.lie_group.regularize(exp)
        return exp

    def exp_from_identity(self, tangent_vec):
        """
        Compute the Riemannian exponential from the identity of the
        Lie group of tangent vector tangent_vec.
        """
        if self.left_or_right == 'left':
            exp = self.left_exp_from_identity(tangent_vec)

        else:
            opp_left_exp = self.left_exp_from_identity(-tangent_vec)

            exp = self.lie_group.inverse(opp_left_exp)

        exp = self.lie_group.regularize(exp)
        return exp

    def exp(self, tangent_vec, base_point):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec.
        """
        base_point = self.lie_group.regularize(base_point)

        jacobian = self.lie_group.jacobian_translation(
                                 base_point,
                                 left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)

        tangent_vec_translated_to_id = np.dot(inv_jacobian, tangent_vec)

        exp_from_id = self.exp_from_identity(
                               tangent_vec_translated_to_id)

        if self.left_or_right == 'left':
            exp = self.lie_group.compose(base_point, exp_from_id)

        else:
            exp = self.lie_group.compose(exp_from_id, base_point)

        exp = self.lie_group.regularize(exp)
        return exp

    def left_log_from_identity(self, point):
        """
        Compute the *left* Riemannian logarithm from the identity of the
        Lie group of tangent vector tangent_vec.

        The left Riemannian logarithm has a special role since the
        left Riemannian logarithm of the canonical metric parameterizes
        the points.
        """
        point = self.lie_group.regularize(point)

        inner_prod_mat = self.inner_product_mat_at_identity
        inv_inner_prod_mat = np.linalg.inv(inner_prod_mat)

        log = np.dot(inv_inner_prod_mat, point)

        return log

    def log_from_identity(self, point):
        """
        Compute the Riemannian logarithm of point at point base_point
        of point for the invariant metric from the identity.
        """
        point = self.lie_group.regularize(point)
        if self.left_or_right == 'left':
            log = self.left_log_from_identity(point)

        else:
            inv_point = self.lie_group.inverse(point)
            left_log = self.left_log_from_identity(inv_point)
            log = - left_log

        return log

    def log(self, point, base_point):
        """
        Compute the Riemannian logarithm of point at point base_point
        of point for the invariant metric.
        """
        base_point = self.lie_group.regularize(base_point)
        point = self.lie_group.regularize(point)

        if self.left_or_right == 'left':
            point_near_id = self.lie_group.compose(
                                   self.lie_group.inverse(base_point),
                                   point)

        else:
            point_near_id = self.lie_group.compose(
                                   point,
                                   self.lie_group.inverse(base_point))

        log_from_id = self.log_from_identity(point_near_id)

        jacobian = self.lie_group.jacobian_translation(
                                       base_point,
                                       left_or_right=self.left_or_right)
        log = np.dot(jacobian, log_from_id)

        return log
