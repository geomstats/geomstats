"""
Left- and right- invariant metrics that exist on Lie groups.
"""

import logging

import geomstats.backend as gs

from geomstats.riemannian_metric import RiemannianMetric


class InvariantMetric(RiemannianMetric):
    """
    Class for:
    - left-invariant metrics
    - right-invariant metrics
    that exist on Lie groups.

    Points are parameterized by the Riemannian logarithm
    for the canonical left-invariant metric.
    """

    def __init__(self, group,
                 inner_product_mat_at_identity=None,
                 left_or_right='left'):
        if inner_product_mat_at_identity is None:
            inner_product_mat_at_identity = gs.eye(self.group.dimension)
        inner_product_mat_at_identity = gs.to_ndarray(
            inner_product_mat_at_identity, to_ndim=3)
        mat_shape = inner_product_mat_at_identity.shape
        assert mat_shape == (1,) + (group.dimension, ) * 2, mat_shape

        assert left_or_right in ('left', 'right')
        eigenvalues = gs.linalg.eigvalsh(inner_product_mat_at_identity)
        mask_pos_eigval = gs.greater(eigenvalues, 0.)
        n_pos_eigval = gs.sum(gs.cast(mask_pos_eigval, gs.int32))
        mask_neg_eigval = gs.less(eigenvalues, 0.)
        n_neg_eigval = gs.sum(gs.cast(mask_neg_eigval, gs.int32))
        mask_null_eigval = gs.isclose(eigenvalues, 0.)
        n_null_eigval = gs.sum(gs.cast(mask_null_eigval, gs.int32))

        self.group = group
        if inner_product_mat_at_identity is None:
            inner_product_mat_at_identity = gs.eye(self.group.dimension)

        self.inner_product_mat_at_identity = inner_product_mat_at_identity
        self.left_or_right = left_or_right
        self.signature = (n_pos_eigval, n_null_eigval, n_neg_eigval)

    def inner_product_at_identity(self, tangent_vec_a, tangent_vec_b):
        """
        Inner product matrix at the tangent space at the identity.
        """
        assert self.group.default_point_type in ('vector', 'matrix')

        if self.group.default_point_type == 'vector':
            tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=2)
            tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=2)

            n_tangent_vec_a = tangent_vec_a.shape[0]
            n_tangent_vec_b = tangent_vec_b.shape[0]

            assert (tangent_vec_a.shape == tangent_vec_b.shape
                    or n_tangent_vec_a == 1
                    or n_tangent_vec_b == 1)

            if n_tangent_vec_a == 1:
                tangent_vec_a = gs.array([tangent_vec_a[0]] * n_tangent_vec_b)

            if n_tangent_vec_b == 1:
                tangent_vec_b = gs.array([tangent_vec_b[0]] * n_tangent_vec_a)

            inner_product_mat_at_identity = gs.array(
                [self.inner_product_mat_at_identity[0]] *
                max(n_tangent_vec_a, n_tangent_vec_b))

            inner_prod = gs.einsum('ij,ijk,ik->i',
                                   tangent_vec_a,
                                   inner_product_mat_at_identity,
                                   tangent_vec_b)

            inner_prod = gs.to_ndarray(inner_prod, to_ndim=2, axis=1)

        elif self.group.default_point_type == 'matrix':
            logging.warning(
                'Only the canonical inner product -Frobenius inner product-'
                ' is implemented for Lie groups whose elements are represented'
                ' by matrices.')
            tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=3)
            tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=3)
            aux_prod = gs.matmul(gs.transpose(tangent_vec_a, axes=(0, 2, 1)),
                                 tangent_vec_b)
            inner_prod = gs.trace(aux_prod)

        return inner_prod

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """
        Inner product between two tangent vectors at a base point.
        """
        if base_point is None:
            return self.inner_product_at_identity(tangent_vec_a,
                                                  tangent_vec_b)
        if self.group.default_point_type == 'vector':
                return super(InvariantMetric, self).inner_product(
                                     tangent_vec_a,
                                     tangent_vec_b,
                                     base_point)

        if self.left_or_right == 'right':
            raise NotImplementedError(
                'inner_product not implemented for right invariant metrics.')
        jacobian = self.group.jacobian_translation(base_point)
        inv_jacobian = gs.linalg.inv(jacobian)
        tangent_vec_a_at_id = gs.matmul(inv_jacobian, tangent_vec_a)
        tangent_vec_b_at_id = gs.matmul(inv_jacobian, tangent_vec_b)
        inner_prod = self.inner_product_at_identity(tangent_vec_a_at_id,
                                                    tangent_vec_b_at_id)
        return inner_prod

    def inner_product_matrix(self, base_point=None):
        """
        Inner product matrix at the tangent space at a base point.
        """
        if self.group.default_point_type == 'matrix':
            raise NotImplementedError(
                'inner_product_matrix not implemented for Lie groups'
                ' whose elements are represented as matrices.')

        if base_point is None:
            base_point = self.group.identity
        base_point = self.group.regularize(base_point)

        jacobian = self.group.jacobian_translation(
                              point=base_point,
                              left_or_right=self.left_or_right)
        assert gs.ndim(jacobian) == 3
        inv_jacobian = gs.linalg.inv(jacobian)
        inv_jacobian_transposed = gs.transpose(inv_jacobian, axes=(0, 2, 1))

        n_base_points = base_point.shape[0]
        inner_product_mat_at_id = gs.array(
            [self.inner_product_mat_at_identity[0]] * n_base_points)

        metric_mat = gs.matmul(inv_jacobian_transposed,
                               inner_product_mat_at_id)
        metric_mat = gs.matmul(metric_mat, inv_jacobian)
        return metric_mat

    def left_exp_from_identity(self, tangent_vec):
        """
        Riemannian exponential of a tangent vector wrt the identity associated
        to the left-invariant metric.

        If the method is called by a right-invariant metric, it uses the
        left-invariant metric associated to the same inner-product matrix
        at the identity.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

        tangent_vec = self.group.regularize_tangent_vec_at_identity(
                                        tangent_vec=tangent_vec,
                                        metric=self)
        sqrt_inner_product_mat = gs.linalg.sqrtm(
            self.inner_product_mat_at_identity)
        mat = gs.transpose(sqrt_inner_product_mat, axes=(0, 2, 1))

        n_tangent_vecs, _ = tangent_vec.shape
        n_mats, _, _ = mat.shape

        if n_mats == 1:
            mat = gs.tile(mat, (n_tangent_vecs, 1, 1))
        if n_tangent_vecs == 1:
            tangent_vec = gs.tile(tangent_vec, (n_mats, 1))
        exp = gs.einsum('ni,nij->nj', tangent_vec, mat)

        exp = self.group.regularize(exp)
        return exp

    def exp_from_identity(self, tangent_vec):
        """
        Riemannian exponential of a tangent vector wrt the identity.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)

        if self.left_or_right == 'left':
            exp = self.left_exp_from_identity(tangent_vec)

        else:
            opp_left_exp = self.left_exp_from_identity(-tangent_vec)
            exp = self.group.inverse(opp_left_exp)

        exp = self.group.regularize(exp)
        return exp

    def exp(self, tangent_vec, base_point=None):
        """
        Riemannian exponential of a tangent vector wrt to a base point.
        """
        if base_point is None:
            base_point = self.group.identity
        base_point = self.group.regularize(base_point)
        if base_point is self.group.identity:
            return self.exp_from_identity(tangent_vec)

        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        assert gs.ndim(tangent_vec) == 2

        n_tangent_vecs, _ = tangent_vec.shape
        n_base_points, _ = base_point.shape

        if n_tangent_vecs == 1:
            tangent_vec = gs.tile(tangent_vec, (n_base_points, 1))
        if n_base_points == 1:
            base_point = gs.tile(base_point, (n_tangent_vecs, 1))

        jacobian = self.group.jacobian_translation(
                                 point=base_point,
                                 left_or_right=self.left_or_right)
        assert gs.ndim(jacobian) == 3
        inv_jacobian = gs.linalg.inv(jacobian)
        inv_jacobian_transposed = gs.transpose(inv_jacobian, axes=(0, 2, 1))
        #print(tangent_vec)
        #print(inv_jacobian_transposed)
        tangent_vec_at_id = gs.einsum('ni,nij->nj',
                                      tangent_vec,
                                      inv_jacobian_transposed)
        exp_from_id = self.exp_from_identity(tangent_vec_at_id)

        if self.left_or_right == 'left':
            exp = self.group.compose(base_point, exp_from_id)

        else:
            exp = self.group.compose(exp_from_id, base_point)

        exp = self.group.regularize(exp)

        return exp

    def left_log_from_identity(self, point):
        """
        Riemannian logarithm of a point wrt the identity associated
        to the left-invariant metric.

        If the method is called by a right-invariant metric, it uses the
        left-invariant metric associated to the same inner-product matrix
        at the identity.
        """
        point = self.group.regularize(point)
        inner_prod_mat = self.inner_product_mat_at_identity
        inv_inner_prod_mat = gs.linalg.inv(inner_prod_mat)
        sqrt_inv_inner_prod_mat = gs.linalg.sqrtm(inv_inner_prod_mat)
        assert sqrt_inv_inner_prod_mat.shape == ((1,)
                                                 + (self.group.dimension,) * 2)
        aux = gs.squeeze(sqrt_inv_inner_prod_mat, axis=0)
        log = gs.matmul(point, aux)
        log = self.group.regularize_tangent_vec_at_identity(
                                             tangent_vec=log,
                                             metric=self)
        assert gs.ndim(log) == 2
        return log

    def log_from_identity(self, point):
        """
        Riemannian logarithm of a point wrt the identity.
        """
        point = self.group.regularize(point)
        if self.left_or_right == 'left':
            log = self.left_log_from_identity(point)

        else:
            inv_point = self.group.inverse(point)
            left_log = self.left_log_from_identity(inv_point)
            log = - left_log

        assert gs.ndim(log) == 2
        return log

    def log(self, point, base_point=None):
        """
        Riemannian logarithm of a point wrt a base point.
        """
        if base_point is None:
            base_point = self.group.identity
        base_point = self.group.regularize(base_point)
        if base_point is self.group.identity:
            return self.log_from_identity(point)

        point = self.group.regularize(point)

        n_points, _ = point.shape
        n_base_points, _ = base_point.shape

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

        n_logs, _ = log_from_id.shape
        n_jacobians, _, _ = jacobian.shape

        if n_logs == 1:
            log_from_id = gs.tile(log_from_id, (n_jacobians, 1))
        if n_jacobians == 1:
            jacobian = gs.tile(jacobian, (n_logs, 1, 1))
        log = gs.einsum('ij,ijk->ik',
                        log_from_id,
                        gs.transpose(jacobian, axes=(0, 2, 1)))
        assert gs.ndim(log) == 2
        return log
