"""Lie groups."""


import geomstats.backend as gs
import geomstats.geometry.riemannian_metric as riemannian_metric
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices


def loss(y_pred, y_true, group, metric=None):
    """Compute loss given by Riemannian metric.

    Parameters
    ----------
    y_pred : array-like, shape=[n_samples, {dim, [n, n]}]
    y_true : array-like, shape=[n_samples, {dim, [n, n]}]
        shape has to match y_pred
    group : LieGroup
    metric : RiemannianMetric, optional
        default: the left invariant metric of the Lie group

    Returns
    -------
    loss : array-like, shape=[n_samples, {dim, [n, n]}]
        the squared (geodesic) distance between y_pred and y_true
    """
    if metric is None:
        metric = group.left_invariant_metric
    metric_loss = riemannian_metric.loss(y_pred, y_true, metric)
    return metric_loss


def grad(y_pred, y_true, group, metric=None):
    """Compute the gradient of the loss function from closed-form expression.

    Parameters
    ----------
    y_pred : array-like, shape=[n_samples, {dim, [n, n]}]
    y_true : array-like, shape=[n_samples, {dim, [n, n]}]
        shape has to match y_pred
    group : LieGroup
    metric : RiemannianMetric, optional
        default: the left invariant metric of the Lie group

    Returns
    -------
    grad : array-like, shape=[n_samples, {dim, [n, n]}]
        tangent vector at point `y_pred`
    """
    if metric is None:
        metric = group.left_invariant_metric
    metric_grad = riemannian_metric.grad(y_pred, y_true, metric)
    return metric_grad


class LieGroup(Manifold):
    """Class for Lie groups.

    In this class, point_type ('vector' or 'matrix') will be used to describe
    the format of the points on the Lie group.
    If point_type is 'vector', the format of the inputs is
    [n_samples, dimension], where dimension is the dimension of the Lie group.
    If point_type is 'matrix' the format of the inputs is
    [n_samples, n, n] where n is the parameter of GL(n) e.g. the amount of rows
    and columns of the matrix.
    """

    def __init__(self, dim, default_point_type='vector', **kwargs):
        super(LieGroup, self).__init__(dim=dim, **kwargs)

        self.left_canonical_metric = InvariantMetric(
            group=self,
            inner_product_mat_at_identity=gs.eye(self.dim),
            left_or_right='left',
        )

        self.right_canonical_metric = InvariantMetric(
            group=self,
            inner_product_mat_at_identity=gs.eye(self.dim),
            left_or_right='right',
        )

        self.metrics = []
        self.default_point_type = default_point_type

    def get_identity(self, point_type=None):
        """Get the identity of the group.

        Parameters
        ----------
        point_type : str, {'matrix', 'vector'}, optional
            default: the default point type

        Returns
        -------
        identity : array-like, shape={[dim], [n, n]}
        """
        raise NotImplementedError(
            'The Lie group identity is not implemented.'
        )

    identity = property(get_identity)

    def compose(self, point_a, point_b, point_type=None):
        """Perform function composition corresponding to the Lie group.

        Multiply the elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, {dim, [n, n]}]
            the left factor in the product
        point_b : array-like, shape=[n_samples, {dim, [n, n]}]
            the right factor in the product
        point_type : str, {'vector', 'matrix'}
            the point_type of the passed point_a and point_b

        Returns
        -------
        composed : array-like, shape=[n_samples, {dim, [n,n]}]
            the product of point_a and point_b along the first dim
        """
        raise NotImplementedError(
            'The Lie group composition is not implemented.'
        )

    @classmethod
    def inverse(cls, point):
        """Compute the inverse law of the Lie group.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dim, [n,n]}]
            the points to be inverted

        point_type : str, {'vector', 'matrix'}, optional
            the point type of the passed point

        Returns
        -------
        inverse : array-like, shape=[n_samples, {dim, [n,n]}]
            the inverted point
        """
        raise NotImplementedError('The Lie group inverse is not implemented.')

    def jacobian_translation(
            self, point, left_or_right='left', point_type=None):
        """Compute the Jacobian of left/right translation by a point.

        Compute the Jacobian matrix of the differential of the left
        translation by the point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dim, [n,n]]
            the points to be inverted

        left_or_right : str, {'left', 'right'}
            indicate whether to calculate the differential of left or right
            translations

        point_type : str, {'vector', 'matrix'}, optional
            default: the default point type
            the point type of the passed point

        Returns
        -------
        jacobian :
            the jacobian of the left/right translation by point
        """
        raise NotImplementedError(
            'The jacobian of the Lie group translation is not implemented.'
        )

    def exp_from_identity(self, tangent_vec, point_type=None):
        """Compute the group exponential of tangent vector from the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dim,[n,n]}]
            the tangent vector to exponentiate
        point_type : str, {'vector', 'matrix'}
            default: the default point type

        Returns
        -------
        point : array-like, shape=[n_samples, {dim,[n,n]}]
        """
        raise NotImplementedError(
            'The group exponential from the identity is not implemented.'
        )

    def exp_not_from_identity(self, tangent_vec, base_point, point_type=None):
        """Calculate the group exponential at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dim,[n,n]}]
        base_point : array-like, shape=[n_samples, {dim,[n,n]}]
        point_type : str, {'vector', 'matrix'}
            default: the default point type

        Returns
        -------
        exp : array-like, shape=[n_samples, {dim,[n,n]}]
            the computed exponential
        """
        if point_type == 'vector':
            jacobian = self.jacobian_translation(
                point=base_point, left_or_right='left', point_type=point_type)
            inv_jacobian = gs.linalg.inv(jacobian)

            tangent_vec_at_id = gs.einsum(
                '...i,...ij->...j',
                tangent_vec, Matrices.transpose(inv_jacobian))
            exp_from_identity = self.exp_from_identity(
                tangent_vec=tangent_vec_at_id, point_type=point_type)
            exp = self.compose(
                base_point, exp_from_identity, point_type=point_type)
            exp = self.regularize(exp, point_type=point_type)
            return exp

        if point_type == 'matrix':
            lie_vec = self.compose(self.inverse(base_point), tangent_vec)
            return self.compose(
                base_point, self.exp_from_identity(lie_vec, point_type))

        raise ValueError('Invalid point_type, expected \'vector\' or '
                         '\'matrix\'')

    def exp(self, tangent_vec, base_point=None, point_type=None):
        """Compute the group exponential at `base_point` of `tangent_vec`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dim,[n,n]}]
        base_point : array-like, shape=[n_samples, {dim,[n,n]}]
            default: self.identity
        point_type : str, {'vector', 'matrix'}
            default: the default point type
            the type of the point

        Returns
        -------
        result : array-like, shape=[n_samples, {dim,[n,n]}]
            The exponentiated tangent vector
        """
        if point_type is None:
            point_type = self.default_point_type
        identity = self.get_identity(point_type=point_type)

        if base_point is None:
            base_point = identity
        base_point = self.regularize(base_point, point_type=point_type)

        if gs.allclose(base_point, identity):
            result = self.exp_from_identity(tangent_vec, point_type=point_type)
        else:
            result = self.exp_not_from_identity(
                tangent_vec, base_point, point_type)
        return result

    def log_from_identity(self, point, point_type=None):
        """Compute the group logarithm of `point` from the identity.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dim,[n,n]}]
        point_type : str, {'vector', 'matrix'}, optional
            defaults to the default point type

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, {dim,[n,n]}]
        """
        raise NotImplementedError(
            'The group logarithm from the identity is not implemented.'
        )

    def log_not_from_identity(self, point, base_point, point_type=None):
        """Compute the group logarithm of `point` from `base_point`.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dim,[n,n]}]
        base_point : array-like, shape=[n_samples, {dim,[n,n]}]
        point_type : str, {'vector', 'matrix'}, optional
            defaults to the default point type

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, {dim,[n,n]}]
        """
        if point_type == 'vector':
            jacobian = self.jacobian_translation(
                point=base_point, left_or_right='left', point_type=point_type)
            point_near_id = self.compose(
                self.inverse(base_point), point, point_type=point_type)
            log_from_id = self.log_from_identity(
                point=point_near_id, point_type=point_type)

            log = gs.einsum(
                '...i,...ij->...j', log_from_id, Matrices.transpose(jacobian))

            return log

        if point_type == 'matrix':
            lie_point = self.compose(self.inverse(base_point), point)
            return self.compose(
                base_point, self.log_from_identity(lie_point, point_type))

        raise ValueError('Invalid point_type, expected \'vector\' or '
                         '\'matrix\'')

    def log(self, point, base_point=None, point_type=None):
        """Compute the group logarithm of `point` relative to `base_point`.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dim,[n,n]}]
        base_point : array-like, shape=[n_samples, {dim,[n,n]}]
        point_type : str, {'vector', 'matrix'}

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, {dim,[n,n]}]
        """
        # TODO(ninamiolane): Build a standalone decorator that *only*
        # deals with point_type None and base_point None
        if point_type is None:
            point_type = self.default_point_type
        identity = self.get_identity(point_type=point_type)
        if base_point is None:
            base_point = identity

        point = self.regularize(point, point_type=point_type)
        base_point = self.regularize(base_point, point_type=point_type)

        if gs.allclose(base_point, identity):
            result = self.log_from_identity(point, point_type=point_type)
        else:
            result = self.log_not_from_identity(point, base_point, point_type)
        return result

    def add_metric(self, metric):
        """Add a metric to the instance's list of metrics."""
        self.metrics.append(metric)

    def lie_bracket(
            self, tangent_vector_a, tangent_vector_b,
            base_point=None, point_type=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : shape=[n_samples, n, n]
        tangent_vector_b : shape=[n_samples, n, n]
        base_point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        bracket : array-like, shape=[n_samples, n, n]
        """
        if point_type is None:
            point_type = self.default_point_type
        if base_point is None:
            base_point = self.get_identity(point_type=point_type)
        inverse_base_point = self.inverse(base_point, point_type=point_type)

        first_term = Matrices.mul(inverse_base_point, tangent_vector_b)
        first_term = Matrices.mul(tangent_vector_a, first_term)

        second_term = Matrices.mul(inverse_base_point, tangent_vector_a)
        second_term = Matrices.mul(tangent_vector_b, second_term)

        return first_term - second_term

    def _is_in_lie_algebra(self, tangent_vec):
        """Check wether a tangent vector belongs to the lie algebra.

        This method could also be in a separate class for the Lie algebra"""
        raise NotImplementedError(
            'The Lie Algebra belongs method is not implemented'
        )

    def is_tangent(self, tangent_vec, base_point=None):
        """Check whether the vector is tangent at base_point."""
        if base_point is None:
            base_point = self.identity

        if gs.allclose(base_point, self.identity):
            tangent_vec_at_id = tangent_vec
        else:
            tangent_vec_at_id = self.compose(
                self.inverse(base_point), tangent_vec)
        is_tangent = self._is_in_lie_algebra(tangent_vec_at_id)
        return is_tangent

    def _to_lie_algebra(self, tangent_vec):
        """Project a vector onto the lie algebra."""
        raise NotImplementedError(
            'The Lie Algebra belongs method is not implemented'
        )

    def to_tangent(self, tangent_vec, base_point=None):
        if base_point is None:
            return self._to_lie_algebra(tangent_vec)
        tangent_vec_at_id = self.compose(
            self.inverse(base_point), tangent_vec)
        regularized = self._to_lie_algebra(tangent_vec_at_id)
        return self.compose(base_point, regularized)
