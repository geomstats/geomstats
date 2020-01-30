"""Lie groups."""


import geomstats.backend as gs
import geomstats.geometry.riemannian_metric as riemannian_metric
from geomstats.geometry.invariant_metric import InvariantMetric
from geomstats.geometry.manifold import Manifold


def loss(y_pred, y_true, group, metric=None):
    """Compute loss given by riemannian metric.

    Parameters
    ----------
    y_pred : array-like, shape=[n_samples, {dimension, [n, n]}]
    y_true : array-like, shape=[n_samples, {dimension, [n, n]}]
        shape has to match y_pred
    group : LieGroup
    metric : RiemannianMetric, optional
        default: the left invariant metric of the Lie group

    Returns
    -------
    loss : array-like, shape=[n_samples, {dimension, [n, n]}]
        the squared (geodesic) distance between y_pred and y_true
    """
    if metric is None:
        metric = group.left_invariant_metric
    loss = riemannian_metric.loss(y_pred, y_true, metric)
    return loss


def grad(y_pred, y_true, group, metric=None):
    """Compute the gradient of the loss function from closed-form expression.

    Parameters
    ----------
    y_pred : array-like, shape=[n_samples, {dimension, [n, n]}]
    y_true : array-like, shape=[n_samples, {dimension, [n, n]}]
        shape has to match y_pred
    group : LieGroup
    metric : RiemannianMetric, optional
        default: the left invariant metric of the Lie group

    Returns
    -------
    grad : array-like, shape=[n_samples, {dimension, [n, n]}]
        tangent vector at point `y_pred`
    """
    if metric is None:
        metric = group.left_invariant_metric
    grad = riemannian_metric.grad(y_pred, y_true, metric)
    return grad


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

    def __init__(self, dimension):
        assert dimension > 0
        Manifold.__init__(self, dimension)

        self.left_canonical_metric = InvariantMetric(
            group=self,
            inner_product_mat_at_identity=gs.eye(self.dimension),
            left_or_right="left",
        )

        self.right_canonical_metric = InvariantMetric(
            group=self,
            inner_product_mat_at_identity=gs.eye(self.dimension),
            left_or_right="right",
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
        identity : array-like, shape={[dimension], [n, n]}
        """
        raise NotImplementedError(
            "The Lie group identity" " is not implemented."
        )

    identity = property(get_identity)

    def compose(self, point_a, point_b, point_type=None):
        """Perform function composition corresponding to the Lie group.

        Multiply the elements `point_a` and `point_b`.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, {dimension, [n, n]}]
            the left factor in the product
        point_b : array-like, shape=[n_samples, {dimension, [n, n]}]
            the right factor in the product
        point_type : str, {'vector', 'matrix'}
            the point_type of the passed point_a and point_b

        Returns
        -------
        composed : array-like, shape=[n_samples, {dimension, [n,n]}]
            the product of point_a and point_b along the first dimension
        """
        raise NotImplementedError(
            "The Lie group composition" " is not implemented."
        )

    def inverse(self, point, point_type=None):
        """Compute the inverse law of the Lie group.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension, [n,n]}]
            the points to be inverted

        point_type : str, {'vector', 'matrix'}, optional
            the point type of the passed point

        Returns
        -------
        inverse : array-like, shape=[n_samples, {dimension, [n,n]}]
            the inverted point
        """
        raise NotImplementedError("The Lie group inverse is not implemented.")

    def jacobian_translation(
        self, point, left_or_right="left", point_type=None
    ):
        """Compute the Jacobian of left/right translation by a point.

        Compute the Jacobian matrix of the differential of the left
        translation by the point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension, [n,n]]
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
            "The jacobian of the Lie group translation is not implemented."
        )

    def exp_from_identity(self, tangent_vec, point_type=None):
        """Compute the group exponential of tangent vector from the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dimension,[n,n]}]
            the tangent vector to exponentiate
        point_type : str, {'vector', 'matrix'}
            default: the default point type

        Returns
        -------
        point : array-like, shape=[n_samples, {dimension,[n,n]}]
        """
        raise NotImplementedError(
            "The group exponential from the identity is not implemented."
        )

    def exp_not_from_identity(self, tangent_vec, base_point, point_type):
        """Calculate the group exponential at base_point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dimension,[n,n]}]
        base_point : array-like, shape=[n_samples, {dimension,[n,n]}]
        point_type : str, {'vector', 'matrix'}
            default: the default point type

        Returns
        -------
        exp : array-like, shape=[n_samples, {dimension,[n,n]}]
            the computed exponential
        """
        jacobian = self.jacobian_translation(
            point=base_point, left_or_right="left", point_type=point_type
        )

        if point_type == "vector":
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            inv_jacobian = gs.linalg.inv(jacobian)

            tangent_vec_at_id = gs.einsum(
                "ni,nij->nj",
                tangent_vec,
                gs.transpose(inv_jacobian, axes=(0, 2, 1)),
            )
            exp_from_identity = self.exp_from_identity(
                tangent_vec=tangent_vec_at_id, point_type=point_type
            )
            exp = self.compose(
                base_point, exp_from_identity, point_type=point_type
            )
            exp = self.regularize(exp, point_type=point_type)
            return exp

        elif point_type == "matrix":
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
            raise NotImplementedError()

    def exp(self, tangent_vec, base_point=None, point_type=None):
        """Compute the group exponential at `base_point` of `tangent_vec`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, {dimension,[n,n]}]
        base_point : array-like, shape=[n_samples, {dimension,[n,n]}]
            default: self.identity
        point_type : str, {'vector', 'matrix'}
            default: the default point type
            the type of the point

        Returns
        -------
        result : array-like, shape=[n_samples, {dimension,[n,n]}]
            The exponentiated tangent vector
        """
        if point_type is None:
            point_type = self.default_point_type

        identity = self.get_identity(point_type=point_type)
        identity = self.regularize(identity, point_type=point_type)
        if base_point is None:
            base_point = identity
        base_point = self.regularize(base_point, point_type=point_type)

        if point_type == "vector":
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
        if point_type == "matrix":
            tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=3)
            base_point = gs.to_ndarray(base_point, to_ndim=3)

        n_tangent_vecs = tangent_vec.shape[0]
        n_base_points = base_point.shape[0]

        assert (
            tangent_vec.shape == base_point.shape
            or n_tangent_vecs == 1
            or n_base_points == 1
        )

        if n_tangent_vecs == 1:
            tangent_vec = gs.array([tangent_vec[0]] * n_base_points)

        if n_base_points == 1:
            base_point = gs.array([base_point[0]] * n_tangent_vecs)

        result = gs.cond(
            pred=gs.allclose(base_point, identity),
            true_fn=lambda: self.exp_from_identity(
                tangent_vec, point_type=point_type
            ),
            false_fn=lambda: self.exp_not_from_identity(
                tangent_vec, base_point, point_type))
        return result

    def log_from_identity(self, point, point_type=None):
        """Compute the group logarithm of `point` from the identity.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension,[n,n]}]
        point_type : str, {'vector', 'matrix'}, optional
            defaults to the default point type

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, {dimension,[n,n]}]
        """
        raise NotImplementedError(
            "The group logarithm from the identity is not implemented."
        )

    def log_not_from_identity(self, point, base_point, point_type):
        """Compute the group logarithm of `point` from `base_point`.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension,[n,n]}]
        base_point : array-like, shape=[n_samples, {dimension,[n,n]}]
        point_type : str, {'vector', 'matrix'}, optional
            defaults to the default point type

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, {dimension,[n,n]}]
        """
        jacobian = self.jacobian_translation(
            point=base_point, left_or_right="left", point_type=point_type
        )
        point_near_id = self.compose(
            self.inverse(base_point), point, point_type=point_type
        )
        log_from_id = self.log_from_identity(
            point=point_near_id, point_type=point_type
        )

        log = gs.einsum(
            "ni,nij->nj",
            log_from_id,
            gs.transpose(jacobian, axes=(0, 2, 1)),
        )

        assert gs.ndim(log) == 2
        return log

    def log(self, point, base_point=None, point_type=None):
        """Compute the group logarithm of `point` relative to `base_point`.

        Parameters
        ----------
        point : array-like, shape=[n_samples, {dimension,[n,n]}]
        base_point : array-like, shape=[n_samples, {dimension,[n,n]}]
        point_type : str, {'vector', 'matrix'}

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, {dimension,[n,n]}]
        """
        if point_type is None:
            point_type = self.default_point_type

        identity = self.get_identity(point_type=point_type)
        if base_point is None:
            base_point = identity

        if point_type == "vector":
            point = gs.to_ndarray(point, to_ndim=2)
            base_point = gs.to_ndarray(base_point, to_ndim=2)
        if point_type == "matrix":
            point = gs.to_ndarray(point, to_ndim=3)
            base_point = gs.to_ndarray(base_point, to_ndim=3)

        point = self.regularize(point, point_type=point_type)
        base_point = self.regularize(base_point, point_type=point_type)

        n_points = point.shape[0]
        n_base_points = base_point.shape[0]

        assert (
            point.shape == base_point.shape
            or n_points == 1
            or n_base_points == 1
        )

        if n_points == 1:
            point = gs.array([point[0]] * n_base_points)

        if n_base_points == 1:
            base_point = gs.array([base_point[0]] * n_points)

        result = gs.cond(
            pred=gs.allclose(base_point, identity),
            true_fn=lambda: self.log_from_identity(
                point, point_type=point_type
            ),
            false_fn=lambda: self.log_not_from_identity(
                point, base_point, point_type
            ),
        )

        return result

    def exponential_barycenter(
        self, points, weights=None, point_type=None
    ):
        """Compute the (weighted) group exponential barycenter of `points`.

        Parameters
        ----------
        points : array-like, shape=[n_samples, {dimension,[n,n]}]
        weights : array-like, shape=[n_samples]
            default is 1 for each point
        point_type : str, {'vector', 'matrix'}

        Returns
        -------
        exp_bar : the exponential_barycenter of the given points
        """
        raise NotImplementedError(
            "The group exponential barycenter is not implemented."
        )

    def add_metric(self, metric):
        """Add a metric to the instance's list of metrics."""
        self.metrics.append(metric)

    def lie_bracket(self, tangent_vector_a, tangent_vector_b, base_point=None):
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
        if base_point is None:
            base_point = self.identity

        base_point = gs.to_ndarray(base_point, to_ndim=3)
        tangent_vector_a = gs.to_ndarray(tangent_vector_a, to_ndim=3)
        tangent_vector_b = gs.to_ndarray(tangent_vector_b, to_ndim=3)

        inverse_base = gs.to_ndarray(
            self.inverse(base_point, point_type="matrix"), to_ndim=3
        )

        first_term = gs.matmul(
            tangent_vector_a, gs.matmul(inverse_base, tangent_vector_b)
        )
        second_term = gs.matmul(
            tangent_vector_b, gs.matmul(inverse_base, tangent_vector_a)
        )
        return first_term - second_term
