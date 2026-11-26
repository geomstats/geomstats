"""Statistical Manifold of Poisson distributions with the Fisher metric.

Lead author: Tra My Nguyen.
"""

from scipy.special import factorial
from scipy.stats import poisson

import geomstats.backend as gs
from geomstats.geometry.positive_reals import PositiveReals, PowerMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)


class PoissonDistributions(InformationManifoldMixin, PositiveReals):
    """Class for the manifold of Poisson distributions.

    This is the parameter space of Poisson distributions
    i.e. the half-line of positive reals.

    Parameters
    ----------
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, equip=True):
        super().__init__(support_shape=(), equip=equip)
        self._scp_rv = PoissonDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return PoissonMetric

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


class PoissonMetric(PowerMetric):
    """Class for the Fisher information metric on Poisson distributions.

    It coincides with the power-Euclidean metric with power 1/2 on the
    positive reals.

    References
    ----------
    .. [AM1981] Atkinson, C., & Mitchell, A. F. (1981). Rao's distance measure.
        Sankhyā: The Indian Journal of Statistics, Series A, 345-365.
    """

    def __init__(self, space):
        super().__init__(space, power=0.5)


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
