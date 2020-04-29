"""Lie groups."""


import geomstats.backend as gs
import geomstats.geometry.riemannian_metric as riemannian_metric
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.matrices import Matrices


ATOL = 1e-8


def loss(y_pred, y_true, group, metric=None):
    """Compute loss given by Riemannian metric.

    Parameters
    ----------
    y_pred : array-like, shape=[..., {dim, [n, n]}]
    y_true : array-like, shape=[..., {dim, [n, n]}]
        shape has to match y_pred
    group : LieGroup
    metric : RiemannianMetric, optional
        default: the left invariant metric of the Lie group

    Returns
    -------
    loss : array-like, shape=[..., {dim, [n, n]}]
        the squared (geodesic) distance between y_pred and y_true
    """
    if metric is None:
        metric = group.left_canonical_metric
    metric_loss = riemannian_metric.loss(y_pred, y_true, metric)
    return metric_loss


def grad(y_pred, y_true, group, metric=None):
    """Compute the gradient of the loss function from closed-form expression.

    Parameters
    ----------
    y_pred : array-like, shape=[..., {dim, [n, n]}]
    y_true : array-like, shape=[..., {dim, [n, n]}]
        shape has to match y_pred
    group : LieGroup
    metric : RiemannianMetric, optional
        default: the left invariant metric of the Lie group

    Returns
    -------
    grad : array-like, shape=[..., {dim, [n, n]}]
        tangent vector at point `y_pred`
    """
    if metric is None:
        metric = group.left_canonical_metric
    metric_grad = riemannian_metric.grad(y_pred, y_true, metric)
    return metric_grad


class LieGroup(Manifold):
    """Class for Lie groups.

    In this class, point_type ('vector' or 'matrix') will be used to describe
    the format of the points on the Lie group.
    If point_type is 'vector', the format of the inputs is
    [..., dimension], where dimension is the dimension of the Lie group.
    If point_type is 'matrix' the format of the inputs is
    [..., n, n] where n is the parameter of GL(n) e.g. the amount of rows
    and columns of the matrix.
    """

    def __init__(self, dim, default_point_type='vector', **kwargs):
        super(LieGroup, self).__init__(
            dim=dim, default_point_type=default_point_type, **kwargs)

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

    def compose(self, point_a, point_b):
        """Perform function composition corresponding to the Lie group.

        Multiply the elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[..., {dim, [n, n]}]
            the left factor in the product
        point_b : array-like, shape=[..., {dim, [n, n]}]
            the right factor in the product

        Returns
        -------
        composed : array-like, shape=[..., {dim, [n, n]}]
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
        point : array-like, shape=[..., {dim, [n, n]}]
            the points to be inverted

        Returns
        -------
        inverse : array-like, shape=[..., {dim, [n, n]}]
            the inverted point
        """
        raise NotImplementedError('The Lie group inverse is not implemented.')

    def jacobian_translation(
            self, point, left_or_right='left'):
        """Compute the Jacobian of left/right translation by a point.

        Compute the Jacobian matrix of the left translation by the point.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]]
            the points to be inverted
        left_or_right : str, {'left', 'right'}
            indicate whether to calculate the differential of left or right
            translations

        Returns
        -------
        jacobian :
            the jacobian of the left/right translation by point
        """
        if self.default_point_type == 'matrix':
            return point

        raise NotImplementedError(
            'The jacobian of the Lie group translation is not implemented.'
        )

    def exp_from_identity(self, tangent_vec):
        """Compute the group exponential of tangent vector from the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
            the tangent vector to exponentiate

        Returns
        -------
        point : array-like, shape=[..., {dim, [n, n]}]
        """
        raise NotImplementedError(
            'The group exponential from the identity is not implemented.'
        )

    def exp_not_from_identity(self, tangent_vec, base_point):
        """Calculate the group exponential at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
        base_point : array-like, shape=[..., {dim, [n, n]}]

        Returns
        -------
        exp : array-like, shape=[..., {dim, [n, n]}]
            the computed exponential
        """
        if self.default_point_type == 'vector':
            jacobian = self.jacobian_translation(
                point=base_point, left_or_right='left')
            inv_jacobian = gs.linalg.inv(jacobian)

            tangent_vec_at_id = gs.einsum(
                '...i,...ij->...j',
                tangent_vec, Matrices.transpose(inv_jacobian))
            exp_from_identity = self.exp_from_identity(
                tangent_vec=tangent_vec_at_id)
            exp = self.compose(
                base_point, exp_from_identity)
            exp = self.regularize(exp)
            return exp

        lie_vec = self.compose(self.inverse(base_point), tangent_vec)
        return self.compose(base_point, self.exp_from_identity(lie_vec))

    def exp(self, tangent_vec, base_point=None):
        """Compute the group exponential at `base_point` of `tangent_vec`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
        base_point : array-like, shape=[..., {dim, [n, n]}]
            default: self.identity

        Returns
        -------
        result : array-like, shape=[..., {dim, [n, n]}]
            The exponentiated tangent vector
        """
        identity = self.get_identity()

        if base_point is None:
            base_point = identity
        base_point = self.regularize(base_point)

        if gs.allclose(base_point, identity):
            result = self.exp_from_identity(tangent_vec)
        else:
            result = self.exp_not_from_identity(
                tangent_vec, base_point)
        return result

    def log_from_identity(self, point):
        """Compute the group logarithm of `point` from the identity.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
        """
        raise NotImplementedError(
            'The group logarithm from the identity is not implemented.'
        )

    def log_not_from_identity(self, point, base_point):
        """Compute the group logarithm of `point` from `base_point`.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
        base_point : array-like, shape=[..., {dim, [n, n]}]

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
        """
        if self.default_point_type == 'vector':
            jacobian = self.jacobian_translation(
                point=base_point, left_or_right='left')
            point_near_id = self.compose(
                self.inverse(base_point), point)
            log_from_id = self.log_from_identity(
                point=point_near_id)

            log = gs.einsum(
                '...i,...ij->...j', log_from_id, Matrices.transpose(jacobian))

            return log

        lie_point = self.compose(self.inverse(base_point), point)
        return self.compose(base_point, self.log_from_identity(lie_point))

    def log(self, point, base_point=None):
        """Compute the group logarithm of `point` relative to `base_point`.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n, n]}]
        base_point : array-like, shape=[..., {dim, [n, n]}]

        Returns
        -------
        tangent_vec : array-like, shape=[..., {dim, [n, n]}]
        """
        # TODO(ninamiolane): Build a standalone decorator that *only*
        # deals with point_type None and base_point None
        identity = self.get_identity(point_type=self.default_point_type)
        if base_point is None:
            base_point = identity

        point = self.regularize(point)
        base_point = self.regularize(base_point)

        if gs.allclose(base_point, identity):
            result = self.log_from_identity(point)
        else:
            result = self.log_not_from_identity(point, base_point)
        return result

    def add_metric(self, metric):
        """Add a metric to the instance's list of metrics."""
        self.metrics.append(metric)

    def lie_bracket(
            self, tangent_vector_a, tangent_vector_b,
            base_point=None):
        """Compute the lie bracket of two tangent vectors.

        For matrix Lie groups with tangent vectors A,B at the same base point P
        this is given by (translate to identity, compute commutator, go back)
        :math:`[A,B] = A_P^{-1}B - B_P^{-1}A`

        Parameters
        ----------
        tangent_vector_a : shape=[..., n, n]
        tangent_vector_b : shape=[..., n, n]
        base_point : array-like, shape=[..., n, n]

        Returns
        -------
        bracket : array-like, shape=[..., n, n]
        """
        if base_point is None:
            base_point = self.get_identity(point_type=self.default_point_type)
        inverse_base_point = self.inverse(base_point)

        first_term = Matrices.mul(inverse_base_point, tangent_vector_b)
        first_term = Matrices.mul(tangent_vector_a, first_term)

        second_term = Matrices.mul(inverse_base_point, tangent_vector_a)
        second_term = Matrices.mul(tangent_vector_b, second_term)

        return first_term - second_term

    def _is_in_lie_algebra(self, tangent_vec, atol=ATOL):
        """Check wether a tangent vector belongs to the lie algebra.

        This method could also be in a separate class for the Lie algebra.

        """
        raise NotImplementedError(
            'The Lie Algebra belongs method is not implemented'
        )

    def is_tangent(self, vector, base_point=None, atol=ATOL):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim_embedding]
            Vector.
        base_point : array-like, shape=[..., dim_embedding]
            Point in the Lie group.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        if base_point is None:
            base_point = self.identity

        if gs.allclose(base_point, self.identity):
            tangent_vec_at_id = vector
        else:
            tangent_vec_at_id = self.compose(
                self.inverse(base_point), vector)
        is_tangent = self._is_in_lie_algebra(tangent_vec_at_id, atol)
        return is_tangent

    def _to_lie_algebra(self, tangent_vec):
        """Project a vector onto the lie algebra."""
        raise NotImplementedError(
            'The Lie Algebra belongs method is not implemented'
        )

    def to_tangent(self, vector, base_point=None):
        """Project a vector onto the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., {dim, [n, n]}]
            Vector to project. Its shape must match the shape of base_point.
        base_point : array-like, shape=[..., {dim, [n, n]}], optional
            Point of the group.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
        """
        if base_point is None:
            return self._to_lie_algebra(vector)
        tangent_vec_at_id = self.compose(
            self.inverse(base_point), vector)
        regularized = self._to_lie_algebra(tangent_vec_at_id)
        return self.compose(base_point, regularized)
