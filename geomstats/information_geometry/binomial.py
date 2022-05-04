"""Statistical Manifold of Binomial distributions with the Fisher metric.

Lead author: Jules Deschamps.
"""

from scipy.stats import binom

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric


class BinomialDistributions(OpenSet):
    """Class for the manifold of binomial distributions.

    This is the parameter space of exponential distributions
    i.e. the half-line of positive reals.
    """

    def __init__(self, n_draws):
        super(BinomialDistributions, self).__init__(
            dim=1,
            ambient_space=Euclidean(dim=1),
            metric=BinomialFisherRaoMetric(n_draws),
        )
        self.n_draws = n_draws

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold of binomial distributions.

        Parameters
        ----------
        point : array-like, shape=[...,]
            Point to be checked.
        atol : float
            Tolerance to evaluate if point belongs to [0,1].
            Optional, default: gs.atol

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents a binomial
            distribution.
        """
        point = gs.cast(gs.array(point), dtype=gs.float32)
        point = gs.to_ndarray(point, 1)
        belongs = gs.logical_and(atol <= point, point <= 1 - atol)
        return gs.squeeze(belongs)

    @staticmethod
    def random_point(n_samples=1):
        """Sample parameters of binomial distributions.

        The uniform distribution on [0, 1] is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[...,]
            Sample of points representing binomial distributions.
        """
        return gs.squeeze(gs.random.rand(n_samples))

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the parameter set.

        The parameter is floored to `gs.atol` if it is negative
        and to '1-gs.atol' if it is greater than 1.

        Parameters
        ----------
        point : array-like, shape=[...,]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[...,]
            Projected point.
        """
        point = gs.cast(gs.array(point), dtype=gs.float32)
        projected = gs.where(
            gs.logical_or(point < atol, point > 1 - atol),
            (1 - atol) * gs.cast((point > 1 - atol), gs.float32)
            + atol * gs.cast((point < atol), gs.float32),
            point,
        )
        return gs.squeeze(projected)

    def sample(self, point, n_samples=1):
        """Sample from the binomial distribution.

        Sample from the binomial distribution with parameter provided by point.

        Parameters
        ----------
        point : array-like, shape=[...]
            Point representing a binomial distribution.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[...]
            Sample from binomial distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=1)
        samples = gs.array([binom.rvs(self.n_draws, point) for i in range(n_samples)])
        return gs.squeeze(gs.transpose(samples))

    def point_to_pmf(self, point):
        """Compute pmf associated to point.

        Compute the probability density function of the binomial
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[...,]
            Point representing a binomial distribution (probability of success).

        Returns
        -------
        pmf : function
            Probability density function of the binomial distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)

        def pmf(k):
            """Generate parameterized function for binomial pmf.

            Parameters
            ----------
            k : array-like, shape=[n_points,]
                Integers in {0, ..., n_draws} at which to
                compute the probability mass function.
            """
            k = gs.array(k, gs.float32)
            k = gs.to_ndarray(k, to_ndim=1)

            pmf_at_k = gs.array(
                [
                    gs.array(binom.pmf(k, self.n_draws, param))
                    for param in gs.to_ndarray(point, 1)
                ]
            )

            return gs.squeeze(pmf_at_k)

        return pmf


class BinomialFisherRaoMetric(RiemannianMetric):
    """Class for the Fisher information metric on binomial distributions.

    References
    ----------
    .. [AM1981] Atkinson, C., & Mitchell, A. F. (1981). Rao's distance measure.
    Sankhyā: The Indian Journal of Statistics, Series A, 345-365.
    """

    def __init__(self, n_draws):
        super(BinomialFisherRaoMetric, self).__init__(dim=1)
        self.n_draws = n_draws

    def squared_dist(self, point_a, point_b, **kwargs):
        """Compute squared distance associated with the binomial Fisher Rao metric.

        Parameters
        ----------
        point_a : array-like, shape=[...,]
            Point representing a binomial distribution (probability of success).
        point_b : array-like, shape=[...,] (same shape as point_a)
            Point representing a binomial distribution (probability of success).

        Returns
        -------
        squared_dist : array-like, shape=[...,]
            Squared distance between points point_a and point_b.
        """
        return (
            4
            * self.n_draws
            * (gs.arcsin(gs.sqrt(point_a)) - gs.arcsin(gs.sqrt(point_b))) ** 2
        )
