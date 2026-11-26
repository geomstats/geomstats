"""Statistical Manifold of exponential distributions with the Fisher metric.

Lead authors: Jules Deschamps, Tra My Nguyen.
"""

from scipy.stats import expon

import geomstats.backend as gs
from geomstats.geometry.positive_reals import LogMetric, PositiveReals
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)


class ExponentialDistributions(InformationManifoldMixin, PositiveReals):
    """Class for the manifold of exponential distributions.

    This is the parameter space of exponential distributions
    i.e. the half-line of positive reals.

    Parameters
    ----------
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, equip=True):
        super().__init__(support_shape=(), equip=equip)
        self._scp_rv = ExponentialDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return ExponentialMetric

    def sample(self, point, n_samples=1):
        """Sample from the exponential distribution.

        Sample from the exponential distribution with parameter provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[...,]
            Point representing an exponential distribution.
        n_samples : int
            Number of points to sample with each parameter in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from exponential distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the exponential
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing an exponential distribution (scale).

        Returns
        -------
        pdf : function
            Probability density function of the exponential distribution with
            scale parameter provided by point.
        """

        def pdf(x):
            """Generate parameterized function for exponential pdf.

            The pdf is parameterized by the scale parameter of the exponential,
            which is equal to 1 / lambda, where lambda is the rate parameter.

            Parameters
            ----------
            x : array-like, shape=[n_points,]
                Points at which to compute the probability density function.

            Returns
            -------
            pdf_at_x : array-like, shape=[..., n_points]
            """
            x = gs.reshape(gs.array(x), (-1,))
            pdf_at_x = point * gs.exp(-point * x)
            return gs.where(x >= 0, pdf_at_x, 0.0)

        return pdf


class ExponentialMetric(LogMetric):
    """Class for the Fisher information metric on exponential distributions.

    It coincides with the logarithmic metric on the positive reals.

    References
    ----------
    .. [AM1981] Atkinson, C., & Mitchell, A. F. (1981). Rao's distance measure.
        Sankhyā: The Indian Journal of Statistics, Series A, 345-365.
    """


class ExponentialDistributionsRandomVariable(ScipyUnivariateRandomVariable):
    """An exponential random variable."""

    def __init__(self, space):
        super().__init__(space, expon.rvs, expon.pdf)

    @staticmethod
    def _flatten_params(point, pre_flat_shape):
        flat_scale_param = gs.reshape(gs.broadcast_to(1 / point, pre_flat_shape), (-1,))
        return {"scale": flat_scale_param}
