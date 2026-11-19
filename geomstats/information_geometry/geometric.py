"""Statistical Manifold of geometric distributions with the Fisher metric.

Lead author: Tra My Nguyen.
"""

from scipy.stats import geom

import geomstats.backend as gs
from geomstats.geometry.open_interval import (
    ArtanhSqrtDiffeo,
    OpenInterval,
    OpenIntervalEuclideanMetric,
)
from geomstats.geometry.positive_reals import PositiveReals
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)


class GeometricDistributions(InformationManifoldMixin, OpenInterval):
    """Class for the manifold of geometric distributions.

    This is the parameter space of geometric distributions
    i.e. the interval (0, 1).

    Parameters
    ----------
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, equip=True):
        super().__init__(
            lower_bound=0.0, upper_bound=1.0, support_shape=(), equip=equip
        )
        self._scp_rv = GeometricDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return GeometricMetric

    def sample(self, point, n_samples=1):
        """Sample from the geometric distribution.

        Sample from the geometric distribution with parameter provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing an geometric distribution.
        n_samples : int
            Number of points to sample with each parameter in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from geometric distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pmf associated to point.

        Compute the probability mass function of the geometric
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point representing an geometric distribution (scale).

        Returns
        -------
        pdf : function
            Probability mass function of the geometric distribution with
            scale parameter provided by point.
        """

        def pmf(k):
            """Generate parameterized function for geometric pmf.

            Compute the probability mass function of the geometric
            distribution with parameters provided by point.

            Parameters
            ----------
            k : array-like, shape=[n_samples,]
                Point in the sample space, i.e. an integer.

            Returns
            -------
            pmf_at_k : array-like, shape=[..., n_samples]
                Probability mass function of the geometric distribution with
                parameters provided by point.
            """
            k = gs.reshape(gs.array(k), (-1,))
            return (1 - point) ** (k - 1) * point

        return pmf


class GeometricMetric(PullbackDiffeoMetric):
    r"""Class for the Fisher information metric on geometric distributions.

    It is the pullback of the Euclidean metric on the half-line
    :math:`(0, \infty)` through :class:`ArtanhSqrtDiffeo`.
    The space is one-dimensional, so the metric is flat.

    References
    ----------
    .. [AM1981] Atkinson, C., & Mitchell, A. F. (1981). Rao's distance measure.
        Sankhyā: The Indian Journal of Statistics, Series A, 345-365.
    """

    def __init__(self, space):
        super().__init__(
            space=space,
            diffeo=ArtanhSqrtDiffeo(),
            image_space=PositiveReals(equip=False).equip_with_metric(
                OpenIntervalEuclideanMetric
            ),
        )

    def metric_matrix(self, base_point):
        """Compute the metric matrix at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point representing a geometric distribution.

        Returns
        -------
        mat : array-like, shape=[..., 1, 1]
            Metric matrix.
        """
        return gs.expand_dims(1 / (base_point**2 * (1 - base_point)), axis=-1)

    def inner_product_derivative_matrix(self, base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        The only entry is
        :math:`\partial_p g_{pp} = (3p - 2) / (p^3 (1 - p)^2)`.

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
        derivative = (3 * base_point - 2) / (base_point**3 * (1 - base_point) ** 2)
        return gs.reshape(derivative, base_point.shape + (1, 1))

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        The only symbol is :math:`\Gamma^p_{pp} = -1 / p + 1 / (2 (1 - p))`.

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
        christoffel = -1 / base_point + 1 / (2 * (1 - base_point))
        return gs.reshape(christoffel, base_point.shape + (1, 1))

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        The only entry is
        :math:`\partial_p \Gamma^p_{pp} = 1 / p^2 + 1 / (2 (1 - p)^2)`.

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
        jac = 1 / base_point**2 + 1 / (2 * (1 - base_point) ** 2)
        return gs.reshape(jac, base_point.shape + (1, 1, 1))


class GeometricDistributionsRandomVariable(ScipyUnivariateRandomVariable):
    """A geometric random variable."""

    def __init__(self, space):
        super().__init__(space, geom.rvs)

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
            geom.pmf(x, point),
        )
