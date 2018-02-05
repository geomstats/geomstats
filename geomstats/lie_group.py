"""Base class for Lie groups."""

import numpy as np

from geomstats.invariant_metric import InvariantMetric
from geomstats.manifold import Manifold


class LieGroup(Manifold):
    """ Base class for Lie groups."""

    def __init__(self, dimension, identity):
        super(LieGroup, self).__init__(dimension)
        self.identity = identity

        self.left_canonical_metric = InvariantMetric(
                    group=self,
                    inner_product_mat_at_identity=np.eye(self.dimension),
                    left_or_right='left')

        self.right_canonical_metric = InvariantMetric(
                    group=self,
                    inner_product_mat_at_identity=np.eye(self.dimension),
                    left_or_right='right')

        self.metrics = []

    def compose(self, point_a, point_b):
        """
        Composition of the Lie group.
        """
        raise NotImplementedError('The Lie group composition'
                                  ' is not implemented.')

    def inverse(self, point):
        """
        Inverse law of the Lie group.
        """
        raise NotImplementedError('The Lie group inverse is not implemented.')

    def jacobian_translation(self, point, left_or_right='left'):
        """
        Compute the jacobian matrix of the differential
    of the left translation by the point.
        """
        raise NotImplementedError(
               'The jacobian of the Lie group translation is not implemented.')

    def group_exp_from_identity(self, tangent_vec):
        """
        Compute the group exponential
        of tangent vector tangent_vec from the identity.
        """
        raise NotImplementedError(
                'The group exponential from the identity is not implemented.')

    def group_exp(self, tangent_vec, base_point=None):
        """
        Compute the group exponential at point base_point
        of tangent vector tangent_vec.
        """
        if base_point is None:
            base_point = self.identity
        base_point = self.regularize(base_point)
        if base_point is self.identity:
            return self.group_exp_from_identity(tangent_vec)

        if tangent_vec.ndim == 1:
            tangent_vec = np.expand_dims(tangent_vec, axis=0)
        assert tangent_vec.ndim == 2

        n_tangent_vecs = tangent_vec.shape[0]
        n_base_points = base_point.shape[0]
        n_exps = np.maximum(n_tangent_vecs, n_base_points)

        assert (n_tangent_vecs == n_base_points
                or n_tangent_vecs == 1
                or n_base_points == 1)

        jacobian = self.jacobian_translation(point=base_point,
                                             left_or_right='left')
        inv_jacobian = np.linalg.inv(jacobian)

        dim = self.dimension
        assert inv_jacobian.shape == (n_base_points, dim, dim)

        tangent_vec_at_id = np.zeros((n_exps, dim))
        for i in range(n_exps):
            inv_jacobian_i = (inv_jacobian[0] if n_base_points == 1
                              else inv_jacobian[i])
            tangent_vec_i = (tangent_vec[0] if n_tangent_vecs == 1
                             else tangent_vec[i])
            tangent_vec_at_id[i] = np.dot(tangent_vec_i,
                                          np.transpose(inv_jacobian_i))

        group_exp_from_identity = self.group_exp_from_identity(
                                       tangent_vec=tangent_vec_at_id)
        group_exp = self.compose(base_point,
                                 group_exp_from_identity)
        group_exp = self.regularize(group_exp)
        return group_exp

    def group_log_from_identity(self, point):
        """
        Compute the group logarithm
        of the point point from the identity.
        """
        raise NotImplementedError(
                'The group logarithm from the identity is not implemented.')

    def group_log(self, point, base_point=None):
        """
        Compute the group logarithm at point base_point
        of the point point.
        """
        if base_point is None:
            base_point = self.identity
        base_point = self.regularize(base_point)
        if base_point is self.identity:
            return self.group_log_from_identity(point)

        point = self.regularize(point)

        n_points = point.shape[0]
        n_base_points = base_point.shape[0]
        n_logs = np.maximum(n_points, n_base_points)

        assert (n_points == n_base_points
                or n_points == 1
                or n_base_points == 1)

        jacobian = self.jacobian_translation(point=base_point,
                                             left_or_right='left')
        point_near_id = self.compose(self.inverse(base_point), point)
        group_log_from_id = self.group_log_from_identity(
                                           point=point_near_id)

        dim = self.dimension
        assert group_log_from_id.shape == (n_logs, dim)
        assert jacobian.shape == (n_base_points, dim, dim)

        group_log = np.zeros((n_logs, dim))
        for i in range(n_logs):
            jacobian_i = jacobian[0] if n_base_points == 1 else jacobian[i]

            log_from_id_i = (group_log_from_id[0] if n_points == 1
                             else group_log_from_id[i])
            group_log[i] = np.dot(log_from_id_i, np.transpose(jacobian_i))

        assert group_log.ndim == 2
        return group_log

    def group_exponential_barycenter(self, points, weights=None):
        """
        Compute the group exponential barycenter.
        """
        raise NotImplementedError(
                'The group exponential barycenter is not implemented.')

    def add_metric(self, metric):
        self.metrics.append(metric)
