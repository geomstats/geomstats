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
                    lie_group=self,
                    inner_product_mat_at_identity=np.eye(self.dimension),
                    left_or_right='left')

        self.right_canonical_metric = InvariantMetric(
                    lie_group=self,
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
            group_exp = self.group_exp_from_identity(tangent_vec)
        else:

            jacobian = self.jacobian_translation(point=base_point,
                                                 left_or_right='left')
            inv_jacobian = np.linalg.inv(jacobian)

            tangent_vec_at_identity = np.dot(inv_jacobian, tangent_vec)
            group_exp_from_identity = self.group_exp_from_identity(
                                           tangent_vec=tangent_vec_at_identity)

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
            group_log = self.group_log_from_identity(point)
        else:
            jacobian = self.jacobian_translation(point=base_point,
                                                 left_or_right='left')
            point_near_id = self.compose(self.inverse(base_point), point)
            group_log_from_id = self.group_log_from_identity(
                                               point=point_near_id)
            group_log = np.dot(jacobian, group_log_from_id)

        return group_log

    def group_exponential_barycenter(self, points, weights=None):
        """
        Compute the group exponential barycenter.
        """
        raise NotImplementedError(
                'The group exponential barycenter is not implemented.')

    def add_metric(self, metric):
        self.metrics.append(metric)
