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

    def __init__(self, group, inner_product_mat_at_identity,
                 left_or_right='left'):
        matrix_shape = inner_product_mat_at_identity.shape
        assert matrix_shape == (group.dimension, group.dimension)
        assert left_or_right in ('left', 'right')

        eigenvalues = np.linalg.eigvalsh(inner_product_mat_at_identity)
        n_pos_eigval = np.sum(eigenvalues > 0)
        n_neg_eigval = np.sum(eigenvalues < 0)
        n_null_eigval = np.sum(eigenvalues == 0)
        assert (n_pos_eigval + n_neg_eigval
                + n_null_eigval) == group.dimension

        self.group = group
        self.inner_product_mat_at_identity = inner_product_mat_at_identity
        self.left_or_right = left_or_right
        self.signature = (n_pos_eigval, n_null_eigval, n_neg_eigval)

    def inner_product_matrix(self, base_point=None):
        """
        Compute the matrix of the Riemmanian metric at point base_point,
        by translating inner_product from the identity to base_point.
        """
        if base_point is None:
            base_point = self.group.identity
        base_point = self.group.regularize(base_point)

        jacobian = self.group.jacobian_translation(
                                  base_point,
                                  left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)
        assert inv_jacobian.ndim == 3
        inv_jacobian_transposed = np.transpose(inv_jacobian, axes=(0, 2, 1))

        inner_product_mat_at_id = self.inner_product_mat_at_identity
        if inner_product_mat_at_id.ndim == 2:
            inner_product_mat_at_id = np.expand_dims(
                             inner_product_mat_at_id,
                             axis=0)
        metric_mat = np.matmul(inv_jacobian_transposed,
                               inner_product_mat_at_id)
        metric_mat = np.matmul(metric_mat, inv_jacobian)
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
        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2
        mat = self.inner_product_mat_at_identity.transpose()
        exp = np.matmul(tangent_vec, mat)

        exp = self.group.regularize(exp)

        return exp

    def exp_from_identity(self, tangent_vec):
        """
        Compute the Riemannian exponential from the identity of the
        Lie group of tangent vector tangent_vec.
        """
        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2

        if self.left_or_right == 'left':
            exp = self.left_exp_from_identity(tangent_vec)

        else:
            opp_left_exp = self.left_exp_from_identity(-tangent_vec)
            exp = self.group.inverse(opp_left_exp)

        exp = self.group.regularize(exp)

        return exp

    def exp_basis(self, tangent_vec, base_point=None):
        """
        Compute the Riemannian exponential at point base_point
        of tangent vector tangent_vec.
        """
        if base_point is None:
            base_point = self.group.identity
        base_point = self.group.regularize(base_point)
        if base_point is self.group.identity:
            return self.exp_from_identity(tangent_vec)

        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2

        n_tangent_vecs, _ = tangent_vec.shape
        n_base_points, _ = base_point.shape

        assert n_tangent_vecs == 1 and n_base_points == 1

        jacobian = self.group.jacobian_translation(
                                 base_point,
                                 left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)

        tangent_vec_at_id = np.matmul(tangent_vec,
                                      np.transpose(inv_jacobian,
                                                   axes=(0, 2, 1)))
        tangent_vec_at_id = np.squeeze(tangent_vec_at_id, axis=0)
        exp_from_id = self.exp_from_identity(tangent_vec_at_id)

        if self.left_or_right == 'left':
            exp = self.group.compose(base_point, exp_from_id)

        else:
            exp = self.group.compose(exp_from_id, base_point)

        exp = self.group.regularize(exp)

        return exp

    def left_log_from_identity(self, point):
        """
        Compute the *left* Riemannian logarithm from the identity of the
        Lie group of tangent vector tangent_vec.

        The left Riemannian logarithm has a special role since the
        left Riemannian logarithm of the canonical metric parameterizes
        the points.
        """
        point = self.group.regularize(point)
        inner_prod_mat = self.inner_product_mat_at_identity
        inv_inner_prod_mat = np.linalg.inv(inner_prod_mat)
        assert inv_inner_prod_mat.shape == (self.group.dimension,
                                            self.group.dimension)
        log = np.dot(point, inv_inner_prod_mat.transpose())

        assert log.ndim == 2
        return log

    def log_from_identity(self, point):
        """
        Compute the Riemannian logarithm of point at point base_point
        of point for the invariant metric from the identity.
        """
        point = self.group.regularize(point)
        if self.left_or_right == 'left':
            log = self.left_log_from_identity(point)

        else:
            inv_point = self.group.inverse(point)
            left_log = self.left_log_from_identity(inv_point)
            log = - left_log

        assert log.ndim == 2
        return log

    def log_basis(self, point, base_point=None):
        """
        Compute the Riemannian logarithm of point at point base_point
        of point for the invariant metric.
        """
        if base_point is None:
            base_point = self.group.identity
        base_point = self.group.regularize(base_point)
        if base_point is self.group.identity:
            return self.log_from_identity(point)

        point = self.group.regularize(point)

        n_points, _ = point.shape
        n_base_points, _ = base_point.shape
        assert n_points == 1 and n_base_points == 1

        if self.left_or_right == 'left':
            point_near_id = self.group.compose(
                                   self.group.inverse(base_point),
                                   point)

        else:
            point_near_id = self.group.compose(
                                   point,
                                   self.group.inverse(base_point))

        log_from_id = self.log_from_identity(point_near_id)

        jacobian = self.group.jacobian_translation(
                                       base_point,
                                       left_or_right=self.left_or_right)

        log = np.matmul(log_from_id, np.transpose(jacobian, axes=(0, 2, 1)))
        log = np.squeeze(log, axis=0)
        assert log.ndim == 2
        return log
