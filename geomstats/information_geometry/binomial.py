"""Statistical Manifold of Binomial distributions with the Fisher metric.

Lead authors: Jules Deschamps, Tra My Nguyen.
"""

import math

from scipy.special import factorial
from scipy.stats import binom

import geomstats.backend as gs
from geomstats.geometry.connection import FlatCoordinateMixin
from geomstats.geometry.open_interval import ArcsinSqrtDiffeo, OpenInterval
from geomstats.geometry.pullback_metric import CoordinatePullbackDiffeoMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)


class BinomialDistributions(InformationManifoldMixin, OpenInterval):
    """Class for the manifold of binomial distributions.

    This is the parameter space of binomial distributions
    i.e. the interval (0, 1).

    Parameters
    ----------
    n_draws : int
        Number of draws.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, n_draws, equip=True):
        self.n_draws = n_draws
        super().__init__(
            lower_bound=0.0, upper_bound=1.0, support_shape=(), equip=equip
        )
        self._scp_rv = BinomialDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return BinomialMetric

    def sample(self, point, n_samples=1):
        """Sample from the binomial distribution.

        Sample from the binomial distribution with parameter provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a binomial distribution.
        n_samples : int
            Number of points to sample with for each parameter in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, dim]
            Sample from binomial distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pmf associated to point.

        Compute the probability mass function of the binomial
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing a binomial distribution (probability of success).

        Returns
        -------
        pmf : function
            Probability mass function of the binomial distribution with
            parameters provided by point.
        """

        def pmf(k):
            """Generate parameterized function for binomial pmf.

            Parameters
            ----------
            k : array-like, shape=[n_samples,]
                Integers in {0, ..., n_draws} at which to
                compute the probability mass function.

            Returns
            -------
            pmf_at_k : array-like, shape=[..., n_samples]
                Values of pdf at k for each value of the parameters provided
                by point.
            """
            k = gs.reshape(gs.array(k), (-1,))
            const = factorial(self.n_draws) / (
                factorial(k) * factorial(self.n_draws - k)
            )
            return (
                gs.from_numpy(const) * (point**k) * ((1 - point) ** (self.n_draws - k))
            )

        return pmf


class BinomialMetric(FlatCoordinateMixin, CoordinatePullbackDiffeoMetric):
    r"""Class for the Fisher information metric on binomial distributions.

    It is the pullback of the Euclidean metric on the interval
    :math:`(0, \pi \sqrt{n})` through :class:`ArcsinSqrtDiffeo`.
    The space is one-dimensional, so the metric is flat.

    References
    ----------
    .. [AM1981] Atkinson, C., & Mitchell, A. F. (1981). Rao's distance measure.
        Sankhyā: The Indian Journal of Statistics, Series A, 345-365.
    """

    def __init__(self, space):
        n_draws = space.n_draws
        super().__init__(
            space=space,
            diffeo=ArcsinSqrtDiffeo(n_draws),
            image_space=OpenInterval(0.0, math.pi * math.sqrt(n_draws)),
        )

    def metric_matrix(self, base_point):
        """Compute the metric matrix at the tangent space at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point representing a binomial distribution.

        Returns
        -------
        mat : array-like, shape=[..., 1, 1]
            Metric matrix.
        """
        return gs.expand_dims(
            self._space.n_draws / (base_point * (1 - base_point)), axis=-1
        )

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        The only entry is
        :math:`\partial_p g_{pp} = n (2p - 1) / (p^2 (1 - p)^2)`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        metric_derivative : array-like, shape=[..., 1, 1, 1]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        derivative = (
            self._space.n_draws
            * (2 * base_point - 1)
            / (base_point**2 * (1 - base_point) ** 2)
        )
        return gs.reshape(derivative, base_point.shape + (1, 1))

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        The only symbol is
        :math:`\Gamma^p_{pp} = -1 / (2 p) + 1 / (2 (1 - p))`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., 1, 1, 1]
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """
        christoffel = -1 / (2 * base_point) + 1 / (2 * (1 - base_point))
        return gs.reshape(christoffel, base_point.shape + (1, 1))

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        The only entry is
        :math:`\partial_p \Gamma^p_{pp} = 1 / (2 p^2) + 1 / (2 (1 - p)^2)`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        jac : array-like, shape=[..., 1, 1, 1, 1]
            Jacobian of the Christoffel symbols.
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`
        """
        jac = 1 / (2 * base_point**2) + 1 / (2 * (1 - base_point) ** 2)
        return gs.reshape(jac, base_point.shape + (1, 1, 1))


class BinomialDistributionsRandomVariable(ScipyUnivariateRandomVariable):
    """A binomial random variable."""

    def __init__(self, space):
        rvs = lambda *args, **kwargs: binom.rvs(space.n_draws, *args, **kwargs)
        super().__init__(space, rvs)

    @staticmethod
    def _flatten_params(point, pre_flat_shape):
        flat_point = gs.reshape(gs.broadcast_to(point, pre_flat_shape), (-1,))
        return {"p": flat_point}

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
            binom.pmf(x, self.space.n_draws, point),
        )
