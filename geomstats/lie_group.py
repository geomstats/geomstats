"""Lie groups."""


import geomstats.backend as gs
import geomstats.riemannian_metric as riemannian_metric

from geomstats.invariant_metric import InvariantMetric
from geomstats.manifold import Manifold


def loss(y_pred, y_true, group, metric=None):
    """
    Loss function given by a riemannian metric.
    """
    if metric is None:
        metric = group.left_invariant_metric
    loss = riemannian_metric.loss(y_pred, y_true, metric)
    return loss


def grad(y_pred, y_true, group, metric=None):
    """
    Closed-form for the gradient of the loss function.

    :return: tangent vector at point y_pred.
    """
    if metric is None:
        metric = group.left_invariant_metric
    grad = riemannian_metric.grad(y_pred, y_true, metric)
    return grad


class LieGroup(Manifold):
    """ Class for Lie groups."""

    def __init__(self, dimension):
        assert dimension > 0
        Manifold.__init__(self, dimension)

        self.left_canonical_metric = InvariantMetric(
                    group=self,
                    inner_product_mat_at_identity=gs.eye(self.dimension),
                    left_or_right='left')

        self.right_canonical_metric = InvariantMetric(
                    group=self,
                    inner_product_mat_at_identity=gs.eye(self.dimension),
                    left_or_right='right')

        self.metrics = []

    def get_identity(self, point_type=None):
        """
        Get the identity of the group.
        """
        raise NotImplementedError('The Lie group identity'
                                  ' is not implemented.')
    identity = property(get_identity)

    def compose(self, point_a, point_b, point_type=None):
        """
        Composition of the Lie group.
        """
        raise NotImplementedError('The Lie group composition'
                                  ' is not implemented.')

    def inverse(self, point, point_type=None):
        """
        Inverse law of the Lie group.
        """
        raise NotImplementedError('The Lie group inverse is not implemented.')

    def jacobian_translation(
            self, point, left_or_right='left', point_type=None):
        """
        Compute the jacobian matrix of the differential
    of the left translation by the point.
        """
        raise NotImplementedError(
               'The jacobian of the Lie group translation is not implemented.')

    def group_exp_from_identity(self, tangent_vec, point_type=None):
        """
        Compute the group exponential
        of tangent vector tangent_vec from the identity.
        """
        raise NotImplementedError(
                'The group exponential from the identity is not implemented.')

    def group_exp(self, tangent_vec, base_point=None, point_type=None):
        """
        Compute the group exponential at point base_point
        of tangent vector tangent_vec.
        """
        if point_type is None:
            point_type = self.default_point_type

        identity = self.get_identity(point_type=point_type)
        identity = self.regularize(identity, point_type=point_type)
        if base_point is None:
            base_point = identity
        base_point = self.regularize(base_point, point_type=point_type)
        if gs.allclose(base_point, identity):
            return self.group_exp_from_identity(
                tangent_vec, point_type=point_type)

        jacobian = self.jacobian_translation(point=base_point,
                                             left_or_right='left',
                                             point_type=point_type)

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            inv_jacobian = gs.linalg.inv(jacobian)

            tangent_vec_at_id = gs.einsum('ij,ijk->ik',
                                          tangent_vec,
                                          gs.transpose(inv_jacobian,
                                                       axes=(0, 2, 1)))
            group_exp_from_identity = self.group_exp_from_identity(
                                           tangent_vec=tangent_vec_at_id,
                                           point_type=point_type)
            group_exp = self.compose(base_point,
                                     group_exp_from_identity,
                                     point_type=point_type)
            group_exp = self.regularize(group_exp, point_type=point_type)
            return group_exp

        elif point_type == 'matrix':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
            raise NotImplementedError()

    def group_log_from_identity(self, point, point_type=None):
        """
        Compute the group logarithm
        of the point point from the identity.
        """
        raise NotImplementedError(
                'The group logarithm from the identity is not implemented.')

    def group_log(self, point, base_point=None, point_type=None):
        """
        Compute the group logarithm at point base_point
        of the point point.
        """
        if point_type is None:
            point_type = self.default_point_type

        identity = self.get_identity(point_type=point_type)
        identity = self.regularize(identity, point_type=point_type)
        if base_point is None:
            base_point = identity
        base_point = self.regularize(base_point, point_type=point_type)
        if gs.allclose(base_point, identity):
            return self.group_log_from_identity(point, point_type=point_type)

        point = self.regularize(point, point_type=point_type)

        jacobian = self.jacobian_translation(point=base_point,
                                             left_or_right='left',
                                             point_type=point_type)
        point_near_id = self.compose(
            self.inverse(base_point), point, point_type=point_type)
        group_log_from_id = self.group_log_from_identity(
                                           point=point_near_id,
                                           point_type=point_type)

        group_log = gs.einsum('ij,ijk->ik',
                              group_log_from_id,
                              gs.transpose(jacobian, axes=(0, 2, 1)))

        assert group_log.ndim == 2
        return group_log

    def group_exponential_barycenter(
           self, points, weights=None, point_type=None):
        """
        Compute the group exponential barycenter.
        """
        raise NotImplementedError(
                'The group exponential barycenter is not implemented.')

    def add_metric(self, metric):
        self.metrics.append(metric)
