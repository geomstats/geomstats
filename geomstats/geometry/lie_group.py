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

    def group_exp_not_from_identity(self, tangent_vec, base_point, point_type):
        jacobian = self.jacobian_translation(
            point=base_point,
            left_or_right='left',
            point_type=point_type)

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            inv_jacobian = gs.linalg.inv(jacobian)

            tangent_vec_at_id = gs.einsum('ni,nij->nj',
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

        if point_type == 'vector':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
        if point_type == 'matrix':
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
            base_point = gs.to_ndarray(base_point, to_ndim=3)

        n_tangent_vecs = tangent_vec.shape[0]
        n_base_points = base_point.shape[0]

        assert (tangent_vec.shape == base_point.shape
                or n_tangent_vecs == 1
                or n_base_points == 1)

        if n_tangent_vecs == 1:
            tangent_vec = gs.array([tangent_vec[0]] * n_base_points)

        if n_base_points == 1:
            base_point = gs.array([base_point[0]] * n_tangent_vecs)

        result = gs.cond(
            pred=gs.allclose(base_point, identity),
            true_fn=lambda: self.group_exp_from_identity(
                tangent_vec, point_type=point_type),
            false_fn=lambda: self.group_exp_not_from_identity(
                tangent_vec, base_point, point_type))
        return result

    def group_log_from_identity(self, point, point_type=None):
        """
        Compute the group logarithm
        of the point point from the identity.
        """
        raise NotImplementedError(
                'The group logarithm from the identity is not implemented.')

    def group_log_not_from_identity(self, point, base_point, point_type):
        jacobian = self.jacobian_translation(point=base_point,
                                             left_or_right='left',
                                             point_type=point_type)
        point_near_id = self.compose(
            self.inverse(base_point), point, point_type=point_type)
        group_log_from_id = self.group_log_from_identity(
                                           point=point_near_id,
                                           point_type=point_type)

        group_log = gs.einsum('ni,nij->nj',
                              group_log_from_id,
                              gs.transpose(jacobian, axes=(0, 2, 1)))

        assert gs.ndim(group_log) == 2
        return group_log

    def group_log(self, point, base_point=None, point_type=None):
        """
        Compute the group logarithm at point base_point
        of the point point.
        """
        if point_type is None:
            point_type = self.default_point_type

        identity = self.get_identity(point_type=point_type)
        if base_point is None:
            base_point = identity

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
        if point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            base_point = gs.to_ndarray(base_point, to_ndim=3)

        point = self.regularize(point, point_type=point_type)
        base_point = self.regularize(base_point, point_type=point_type)

        n_points = point.shape[0]
        n_base_points = base_point.shape[0]

        assert (point.shape == base_point.shape
                or n_points == 1
                or n_base_points == 1)

        if n_points == 1:
            point = gs.array([point[0]] * n_base_points)

        if n_base_points == 1:
            base_point = gs.array([base_point[0]] * n_points)

        result = gs.cond(
            pred=gs.allclose(base_point, identity),
            true_fn=lambda: self.group_log_from_identity(
                point, point_type=point_type),
            false_fn=lambda: self.group_log_not_from_identity(
                point, base_point, point_type))

        return result

    def group_exponential_barycenter(
           self, points, weights=None, point_type=None):
        """
        Compute the group exponential barycenter.
        """
        raise NotImplementedError(
                'The group exponential barycenter is not implemented.')

    def add_metric(self, metric):
        self.metrics.append(metric)
