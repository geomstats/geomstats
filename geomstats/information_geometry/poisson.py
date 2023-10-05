"""Statistical Manifold of Poisson distributions with the Fisher metric.

Lead author: Tra My Nguyen.
"""

from scipy.special import factorial
from scipy.stats import poisson

import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)


class PoissonDistributions(InformationManifoldMixin, OpenSet):
    """Class for the manifold of Poisson distributions.

    This is the parameter space of Poisson distributions
    i.e. [the half-line of positive reals].
    """

    def __init__(self, equip=True):
        super().__init__(
            dim=1,
            embedding_space=Euclidean(dim=1, equip=False),
            support_shape=(),
            equip=equip,
        )
        self._scp_rv = PoissonDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PoissonMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold of Poisson distributions.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point to be checked.
        atol : float
            Tolerance to evaluate positivity.
            Optional, default: gs.atol

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents an Poisson
            distribution.
        """
        belongs_shape = self.shape == point.shape[-self.point_ndim :]
        if not belongs_shape:
            shape = point.shape[: -self.point_ndim]
            return gs.zeros(shape, dtype=bool)
        return gs.squeeze(point >= atol)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample parameters of Possion distributions.

        The uniform distribution on (0, bound) is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Right-end ot the segment where Poisson parameters are sampled.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[n_samples,]
            Sample of points representing Poisson distributions.
        """
        size = (n_samples, self.dim) if n_samples != 1 else (self.dim,)
        return bound * gs.random.rand(*size)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        Return a point belonging to the half-line of positive reals
        within the given tolerance.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[..., 1]
            Projected point.
        """
        return gs.where(point < atol, atol, point)

    def sample(self, point, n_samples=1):
        """Sample from the Poisson distribution.

        Sample from the Poisson distribution with parameter provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing an Poisson distribution.
        n_samples : int
            Number of points to sample with each parameter in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from Poisson distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pmf associated to point.

        Compute the probability mass function of the Poisson
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing an Poisson distribution (scale).

        Returns
        -------
        pdf : function
            Probability mass function of the Poisson distribution with
            scale parameter provided by point.
        """

        def pmf(k):
            """Generate parameterized function for Poisson pmf.

            Compute the probability mass function of the Poisson
            distribution with parameters provided by point.

            Parameters
            ----------
            k : array-like, shape=[n_samples,]
                Point in the sample space, i.e. an integer.

            Returns
            -------
            pmf_at_k : array-like, shape=[..., n_samples]
                Probability mass function of the Poisson distribution with
                parameters provided by point.
            """
            k = gs.reshape(gs.array(k), (-1,))
            return point**k * gs.exp(-point) / gs.from_numpy(factorial(k))

        return pmf


class PoissonMetric(RiemannianMetric):
    """Class for the Fisher information metric on Poisson distributions.

    References
    ----------
    .. [AM1981] Atkinson, C., & Mitchell, A. F. (1981). Rao's distance measure.
        Sankhyā: The Indian Journal of Statistics, Series A, 345-365.
    """

    def squared_dist(self, point_a, point_b):
        """Compute squared distance associated with the Poisson metric.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point representing an Poisson distribution (lambda parameter).
        point_b : array-like, shape=[..., 1]
            Point representing a Poisson distribution (lambda parameter).

        Returns
        -------
        squared_dist : array-like, shape=[...,]
            Squared distance between points point_a and point_b.
        """
        return gs.squeeze(4 * (point_a - 2 * gs.sqrt(point_a * point_b) + point_b))

    def metric_matrix(self, base_point):
        """Compute the metric matrix at the tangent space at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point representing a Poisson distribution.

        Returns
        -------
        mat : array-like, shape=[..., 1, 1]
            Metric matrix.
        """
        return gs.expand_dims(1 / base_point, axis=-1)

    @staticmethod
    def _geodesic_path(t, constant_a, constant_b):
        """Generate parameterized function for geodesic curve.

        Parameters
        ----------
        t : array-like, shape=[n_times,]
            Times at which to compute points of the geodesics.

        Returns
        -------
        geodesic : array-like, shape=[..., n_times, 1]
            Values of the geodesic at times t.
        """
        return gs.expand_dims((constant_a * t + constant_b) ** 2, axis=-1)

    def _geodesic_ivp(self, initial_point, initial_tangent_vec):
        """Solve geodesic initial value problem.

        Compute the parameterized function for the geodesic starting at
        initial_point with initial velocity given by initial_tangent_vec.

        Parameters
        ----------
        initial_point : array-like, shape=[..., 1]
            Initial point.

        initial_tangent_vec : array-like, shape=[..., 1]
            Tangent vector at initial point.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point with velocity initial_tangent_vec.
        """
        constant_a = initial_tangent_vec / (2 * gs.sqrt(initial_point))
        constant_b = gs.sqrt(initial_point)
        return lambda t: self._geodesic_path(t, constant_a, constant_b)

    def _geodesic_bvp(self, initial_point, end_point):
        """Solve geodesic boundary problem.

        Compute the parameterized function for the geodesic starting at
        initial_point and ending at end_point.

        Parameters
        ----------
        initial_point : array-like, shape=[..., 1]
            Initial point.
        end_point : array-like, shape=[..., 1]
            End point.

        Returns
        -------
        path : function
            Parameterized function for the geodesic curve starting at
            initial_point and ending at end_point.
        """
        constant_a = gs.sqrt(end_point) - gs.sqrt(initial_point)
        constant_b = gs.sqrt(initial_point)
        return lambda t: self._geodesic_path(t, constant_a, constant_b)

    def exp(self, tangent_vec, base_point):
        """Compute exp map of a base point in tangent vector direction.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        exp : array-like, shape=[..., 1]
            End point of the geodesic starting at base_point with
            initial velocity tangent_vec.
        """
        return (tangent_vec / (2 * gs.sqrt(base_point)) + gs.sqrt(base_point)) ** 2

    def log(self, point, base_point):
        """Compute log map using a base point and an end point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            End point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Initial velocity of the geodesic starting at base_point and
            reaching point at time 1.
        """
        return 2 * gs.sqrt(base_point) * (gs.sqrt(point) - gs.sqrt(base_point))


class PoissonDistributionsRandomVariable(ScipyUnivariateRandomVariable):
    """A Poisson random variable."""

    def __init__(self, space):
        super().__init__(space, poisson.rvs)

    @staticmethod
    def _flatten_params(point, pre_flat_shape):
        flat_point = gs.reshape(gs.broadcast_to(point, pre_flat_shape), (-1,))
        return {"mu": flat_point}

    def pdf(self, x, point):
        """Evaluate the probability density function at x.

        Parameters
        ----------
        x : array-like, shape=[n_samples, *support_shape]
            Sample points at which to compute the probability density function.
        point : array-like, shape=[..., *space.shape]
            Point representing a distribution.

        Returns
        -------
        pdf_at_x : array-like, shape=[..., n_samples, *support_shape]
            Values of pdf at x for each value of the parameters provided
            by point.
        """
        return gs.from_numpy(
            poisson.pmf(x, point),
        )
