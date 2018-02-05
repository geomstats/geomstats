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

    def inner_product_matrix(self, base_point):
        """
        Compute the matrix of the Riemmanian metric at point base_point,
        by translating inner_product from the identity to base_point.
        """
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

    def exp(self, tangent_vec, base_point=None):
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

        n_tangent_vecs = tangent_vec.shape[0]
        n_base_points = base_point.shape[0]
        n_exps = np.maximum(n_tangent_vecs, n_base_points)

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        jacobian = self.group.jacobian_translation(
                                 base_point,
                                 left_or_right=self.left_or_right)
        inv_jacobian = np.linalg.inv(jacobian)

        dim = self.group.dimension
        assert inv_jacobian.shape == (n_base_points, dim, dim)

        tangent_vec_at_id = np.zeros((n_exps, dim))
        for i in range(n_exps):
            inv_jacobian_i = (inv_jacobian[0] if n_base_points == 1
                              else inv_jacobian[i])
            tangent_vec_i = (tangent_vec[0] if n_tangent_vecs == 1
                             else tangent_vec[i])
            tangent_vec_at_id[i] = np.dot(tangent_vec_i,
                                          np.transpose(inv_jacobian_i))

        exp_from_id = self.exp_from_identity(
                               tangent_vec_at_id)

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

    def log(self, point, base_point=None):
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

        n_points = point.shape[0]
        n_base_points = base_point.shape[0]
        n_logs = np.maximum(n_points, n_base_points)

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

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
        dim = self.group.dimension
        assert log_from_id.shape == (n_logs, dim)
        assert jacobian.shape == (n_base_points, dim, dim)

        log = np.zeros((n_logs, dim))
        for i in range(n_logs):
            jacobian_i = jacobian[0] if n_base_points == 1 else jacobian[i]

            log_from_id_i = log_from_id[0] if n_points == 1 else log_from_id[i]
            log[i] = np.dot(log_from_id_i, np.transpose(jacobian_i))

        assert log.ndim == 2
        return log
