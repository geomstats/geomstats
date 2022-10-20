"""The real positive axis.

The real positive axis endowed with the Information geometry metric.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-16


class PositiveReals(OpenSet):
    """Class for the positive real axis.

    The real positive axis endowed with the Information geometry metric.
    """

    def __init__(self, scale=1, **kwargs):
        """Construct the positive real axis."""
        super().__init__(
            dim=1,
            ambient_space=Euclidean(1),
            metric=PositiveRealsMetric(scale=scale),
            **kwargs
        )
        self.scale = scale

    def projection(self, point):
        """Project a point on the real axis on manifold.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point in ambient manifold.

        Returns
        -------
        projected : array-like, shape=[..., 1]
            Projected point.
        """
        return gs.where(point <= 0, 2 * gs.atol, point)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the positive real axis.

        Evaluate if a point belongs to the positive real axis,
        i.e. evaluate if its norm is lower than one.

        Parameters
        ----------
        point : array-like, shape=[...,]
                Input points.
        atol : float,
            Optional, default: gs.atol

        Returns
        -------
        belongs : array-like, shape=[...,]
        """
        is_real = point == point.real
        is_positive = point > 0
        belongs = is_real * is_positive
        return belongs


class PositiveRealsMetric(RiemannianMetric):
    """Class for the positive real metric."""

    def __init__(self, scale=1):
        """Construct the positive real metric.

        The positive real axis is considered as a real space and its
        dimension and signature are defined accordingly.
        """
        self.dim = 1
        self.signature = (1, 0, 0)
        assert scale > 0, "The scale should be strictly positive"
        self.scale = scale
        self.point_shape = (1,)
        self.n_dim_point = 1

    def inner_product_matrix(self, base_point):
        """Compute inner product matrix at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[n_samples, 1], optional

        Returns
        -------
        matrices : array-like of shape (n_samples, 1)
        """
        matrix = 1 / base_point**2
        matrix *= self.scale**2
        return matrix

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the inner product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, 1]

        tangent_vec_b : array-like, shape=[n_samples, 1]

        base_point : array-like, shape (n_samples, 1)

        Returns
        -------
        inner_prod : array, shape=[n_samples, 1]
        """
        inner_product_matrix = self.inner_product_matrix(base_point=base_point)
        inner_prod = tangent_vec_a * inner_product_matrix * tangent_vec_b
        return inner_prod

    def squared_norm(self, vector, base_point):
        """Compute the squared norm of a vector at a given base point.

        Squared norm of a vector associated with the inner product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, 1]

        base_point : array-like, shape=[n_samples, 1]

        Returns
        -------
        sq_norm : array-like, shape=[n_samples, 1]
        """
        sq_norm = self.inner_product(vector, vector, base_point=base_point)
        return gs.real(sq_norm)

    def exp(self, tangent_vec, base_point):
        """Compute Riemannian exponential of tangent vector wrt to base point.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, 1]

        base_point : array-like, shape=[n_samples, 1]

        Returns
        -------
        exp : array-like, shape=[n_samples, 1]
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        exp = base_point * gs.exp(tangent_vec / base_point)
        return exp

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, 1]

        base_point : array-like, shape=[n_samples, 1]

        Returns
        -------
        log : array-like, shape=[n_samples, dimension + 1]
        """
        point = gs.to_ndarray(point, to_ndim=2)
        log = base_point * gs.log(point / base_point)
        return log

    def squared_dist(self, point_a, point_b, epsilon=EPSILON):
        """Compute the squared distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, 1]

        point_b : array-like, shape=[n_samples, 1]

        epsilon : int, positive constant
            Security that can be used not to divide by zero.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
        """
        sq_dist = gs.log(point_b / point_a) ** 2
        sq_dist *= self.scale**2
        return gs.real(sq_dist)

    def dist(self, point_a, point_b, epsilon=EPSILON):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, 1]

        point_b : array-like, shape=[n_samples, 1]

        epsilon : int, positive constant
            Security that can be used not to divide by zero.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
        """
        dist = gs.abs(gs.log(point_b / point_a))
        dist *= self.scale
        return gs.real(dist)
