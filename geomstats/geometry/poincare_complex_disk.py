"""The Poincare complex disk.

The Poincare complex disk is a representation of the Hyperbolic space of dimension 2.

Lead author: Yann Cabanes.
"""

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.hermitian import Hermitian
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6

EPSILON = 1e-16


class PoincareComplexDisk(OpenSet):
    """Class for the Poincare disk.

    The Poincare disk is a representation of the Hyperbolic
    space of dimension 2.
    Its complex dimension is 1.
    """

    def __init__(self, scale=1, **kwargs):
        """Construct the Poincare disk."""
        super().__init__(
            dim=1,
            ambient_space=Hermitian(dim=1),
            metric=PoincareComplexDiskMetric(scale=scale),
            **kwargs)
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
        return gs.where(gs.abs(point) >= 1, (1 - 2 * gs.atol) / gs.abs(point) * point, point)

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the Poincare complex disk.

        Evaluate if a point belongs to the Poincare complex disk,
        i.e. evaluate if its squared is lower than one.

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
        belongs = gs.abs(point) < 1
        return belongs


class PoincareComplexDiskMetric(RiemannianMetric):
    """Class for the Poincare metric."""

    def __init__(self, scale=1):
        """Construct the Poincare metric.

        The Poincare Disk is here considered as a complex space and its
        dimension and signature are defined accordingly.
        """
        self.dim = 1
        self.signature = (1, 0, 0)
        assert scale > 0, 'The scale should be strictly positive'
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
        matrix = 1 / (1 - gs.abs(base_point) ** 2) ** 2
        matrix *= self.scale ** 2
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
        inner_product_matrix = self.inner_product_matrix(
            base_point=base_point)
        inner_prod = \
            gs.conj(tangent_vec_a) * inner_product_matrix * tangent_vec_b
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
        sq_norm = self.inner_product(
            vector,
            vector,
            base_point=base_point)
        sq_norm = gs.real(sq_norm)
        return sq_norm

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
        data_type = (tangent_vec + base_point).dtype
        data_is_complex = (data_type == 'complex')
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        theta = gs.angle(tangent_vec, deg=False)
        s = 2 * gs.abs(tangent_vec) / (1 - gs.abs(base_point) ** 2)
        exp = (base_point + gs.exp(1j * theta) +
               (base_point - gs.exp(1j * theta)) *
               gs.exp(-s)) / (
                    1 + gs.conj(base_point) *
                    gs.exp(1j * theta) +
                    (1 - gs.conj(base_point) *
                     gs.exp(1j * theta)) * gs.exp(-s))
        # exp = gs.array(exp, dtype=data_type)
        if not data_is_complex:
            exp = exp.real
        return exp

    @staticmethod
    def tau(point_a, point_b, epsilon=EPSILON, type='float'):
        """Compute the distance in the Poincare disk of scale 1."""
        num = gs.abs(point_a - point_b)
        den = gs.abs(1 - point_a * gs.conj(point_b))
        den = gs.maximum(den, epsilon)
        delta = num / den
        delta = gs.minimum(delta, 1 - epsilon)
        result = (1 / 2) * gs.log((1 + delta) / (1 - delta))
        result = gs.array(result, dtype=type)
        return result

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
        data_type = (point + base_point).dtype
        data_is_complex = (data_type == 'complex')
        point = gs.to_ndarray(point, to_ndim=2)
        log = self.tau(base_point, point, type=complex)
        angle = gs.angle(point - base_point) - gs.angle(
            1 - gs.conj(base_point) * point)
        log *= gs.exp(1j * angle)
        log *= (1 - gs.abs(base_point) ** 2)
        if not data_is_complex:
            log = gs.real(log)
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
        dist = self.tau(point_a, point_b, epsilon=epsilon)
        sq_dist = dist ** 2
        sq_dist *= self.scale ** 2
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
        dist = self.tau(point_a, point_b, epsilon=epsilon)
        dist *= self.scale
        return gs.real(dist)
