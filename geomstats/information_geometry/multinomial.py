"""Statistical Manifold of multinomial distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

from scipy.stats import multinomial

import geomstats.backend as gs
from geomstats.geometry.positive_hypersphere import PositiveHypersphere
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.simplex import OpenSimplex, SqrtDiffeo
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyMultivariateRandomVariable,
)


class MultinomialDistributions(InformationManifoldMixin, OpenSimplex):
    r"""Class for the manifold of multinomial distributions.

    This is the set of ``n+1``-tuples of positive reals that sum up to one,
    i.e. the ``n``-simplex. Each point is the parameter of a multinomial
    distribution, i.e. gives the probabilities of ``n`` different outcomes
    in a single experiment.

    Parameters
    ----------
    dim : int
        Dimension of the parameter manifold of multinomial distributions.
        The number of outcomes is dim + 1.
    n_draws : int
        Number of draws.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, dim, n_draws, equip=True):
        self.n_draws = n_draws
        super().__init__(dim=dim, support_shape=(dim + 1,))
        if equip:
            self.equip_with_metric()

        self._scp_rv = MultinomialRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MultinomialMetric

    def sample(self, point, n_samples=1):
        """Sample from the multinomial distribution.

        Sample from the multinomial distribution with parameters provided by
        point. This gives samples in the simplex.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Parameters of a multinomial distribution, i.e. probabilities
            associated to dim + 1 outcomes.
        n_samples : int
            Number of points to sample with each set of parameters in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, dim + 1]
            Samples from multinomial distributions.
            Note that this can be of shape [n_points, n_samples, dim + 1] if
            several points and several samples are provided as inputs.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the Multinomial
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a beta distribution.

        Returns
        -------
        pdf : function
            (Discrete) probability density function.
        """
        return lambda x: self._scp_rv.pdf(x, point=point)


class MultinomialMetric(PullbackDiffeoMetric):
    """Class for the Fisher information metric on multinomial distributions.

    The Fisher information metric on the ``n``-simplex of multinomial
    distributions parameters can be obtained as the pullback metric of the
    ``n``-sphere using the componentwise square root.

    References
    ----------
    .. [K1989] R. E. Kass. The Geometry of Asymptotic Inference. Statistical
        Science, 4(3): 188 - 234, 1989.
    """

    def __init__(self, space):
        super().__init__(
            space,
            diffeo=SqrtDiffeo(),
            image_space=PositiveHypersphere(dim=space.dim).equip_with_metric(
                ScalarProductMetric, scale=(2 * gs.sqrt(space.n_draws)) ** 2
            ),
        )


class MultinomialRandomVariable(ScipyMultivariateRandomVariable):
    """A multinomial random variable."""

    def __init__(self, space):
        rvs = lambda *args, **kwargs: multinomial.rvs(space.n_draws, *args, **kwargs)
        pdf = lambda x, *args, **kwargs: multinomial.pmf(
            x, space.n_draws, *args, **kwargs
        )
        super().__init__(space, rvs, pdf)
