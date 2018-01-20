"""Base class for Lie groups."""

import numpy as np

from geomstats.invariant_metric import InvariantMetric
from geomstats.manifold import Manifold


class LieGroup(Manifold):
    """ Base class for Lie groups."""

    def __init__(self, dimension, identity):
        Manifold.__init__(self, dimension)
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

    def group_exp(self, tangent_vec, base_point=None):
        """
        Compute the group exponential at point base_point
        of tangent vector tangent_vec.
        """
        if base_point is None:
            return self.group_exp(tangent_vec, self.identity)
        else:
            raise NotImplementedError(
                'The group exponential is not implemented.')

    def group_log(self, point, base_point=None):
        """
        Compute the group logarithm at point base_point
        of the point point.
        """
        if base_point is None:
            return self.group_log(point, self.identity)
        else:
            raise NotImplementedError(
                'The group logarithm is not implemented.')

    def add_metric(self, metric):
        self.metrics.append(metric)
