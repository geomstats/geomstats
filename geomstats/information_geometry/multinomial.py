"""Statistical Manifold of multinomial distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

from scipy.stats import dirichlet, multinomial

import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyMultivariateRandomVariable,
)
from geomstats.vectorization import repeat_out


class SimplexToPositiveHypersphere(Diffeo):
    """Diffeomorphism between simplex and its image under componentwise square root."""

    @staticmethod
    def __call__(point):
        """Send point of the simplex to the sphere.

        The map takes the square root of each component.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on the simplex.

        Returns
        -------
        image_point : array-like, shape=[..., dim]
            Point on the sphere.
        """
        return point ** (1 / 2)

    @staticmethod
    def inverse(image_point):
        """Send point of the sphere to the simplex.

        The map squares each component.

        Parameters
        ----------
        image_point : array-like, shape=[..., dim]
            Point on the sphere.

        Returns
        -------
        point : array-like, shape=[..., dim]
            Point on the simplex.
        """
        return image_point**2

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Send tangent vector of the simplex to tangent space of sphere.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vec to the simplex at base point.
        base_point : array-like, shape=[..., dim]
            Point of the simplex.
        image_point : array-like, shape=[..., dim]
            Point of the sphere.

        Returns
        -------
        image_tangent_vector : array-like, shape=[..., dim]
            Tangent vec to the sphere at the image of
            base point by simplex_to_sphere.
        """
        if image_point is None:
            image_point = self(base_point)

        return gs.einsum("...i,...i->...i", tangent_vec, 1 / (2 * image_point))

    def inverse_tangent(self, tangent_vec, image_point=None, base_point=None):
        """Send tangent vector of the sphere to tangent space of simplex.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vec to the sphere at base point.
        image_point : array-like, shape=[..., dim]
            Point of the sphere.
        base_point : array-like, shape=[..., dim]
            Point of the simplex.

        Returns
        -------
        tangent_vec_simplex : array-like, shape=[..., dim]
            Tangent vec to the simplex at the image of
            base point by sphere_to_simplex.
        """
        if image_point is None:
            image_point = self(base_point)

        return gs.einsum("...i,...i->...i", tangent_vec, 2 * image_point)


class MultinomialDistributions(InformationManifoldMixin, LevelSet):
    r"""Class for the manifold of multinomial distributions.

    This is the set of `n+1`-tuples of positive reals that sum up to one,
    i.e. the `n`-simplex. Each point is the parameter of a multinomial
    distribution, i.e. gives the probabilities of `n` different outcomes
    in a single experiment.

    Attributes
    ----------
    dim : int
        Dimension of the parameter manifold of multinomial distributions.
        The number of outcomes is dim + 1.
    embedding_manifold : Manifold
        Embedding manifold.
    """

    def __init__(self, dim, n_draws, equip=True):
        self.dim = dim
        self.n_draws = n_draws

        super().__init__(
            dim=dim, support_shape=(dim + 1,), shape=(dim + 1,), equip=equip
        )
        self._scp_rv = MultinomialRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MultinomialMetric

    def _define_embedding_space(self):
        return Euclidean(self.dim + 1)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]

        Returns
        -------
        submersed_point : array-like, shape=[...]
        """
        return gs.sum(point, axis=-1) - 1.0

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
        point : Ignored.

        Returns
        -------
        submersed_vector : array-like, shape=[...]
        """
        return gs.sum(vector, axis=-1)

    def random_point(self, n_samples=1):
        """Generate parameters of multinomial distributions.

        The Dirichlet distribution on the simplex is used
        to generate parameters.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Sample of points representing multinomial distributions.
        """
        samples = gs.from_numpy(dirichlet.rvs(gs.ones(self.dim + 1), size=n_samples))

        return samples[0] if n_samples == 1 else samples

    def projection(self, point, atol=gs.atol):
        """Project a point on the simplex.

        Negative components are replaced by zero and the point is
        renormalized by its 1-norm.

        Parameters
        ----------
        point: array-like, shape=[..., dim + 1]
            Point in embedding Euclidean space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point projected on the simplex.
        """
        point_quadrant = gs.where(point < atol, atol, point)
        norm = gs.sum(point_quadrant, axis=-1)
        projected_point = gs.einsum("...,...i->...i", 1.0 / norm, point_quadrant)
        return projected_point

    def to_tangent(self, vector, base_point=None):
        """Project a vector to the tangent space.

        Project a vector in Euclidean space on the tangent space of
        the simplex at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector in Euclidean space.
        base_point : array-like, shape=[..., dim + 1]
            Point on the simplex defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        vector : array-like, shape=[..., dim + 1]
            Tangent vector in the tangent space of the simplex
            at the base point.
        """
        component_mean = gs.mean(vector, axis=-1)
        tangent_vec = gs.transpose(gs.transpose(vector) - component_mean)

        return repeat_out(
            self.point_ndim, tangent_vec, vector, base_point, out_shape=self.shape
        )

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
            Optional, default: 1.

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

    The Fisher information metric on the `n`-simplex of multinomial
    distributions parameters can be obtained as the pullback metric of the
    `n`-sphere using the componentwise square root.

    References
    ----------
    .. [K2003] R. E. Kass. The Geometry of Asymptotic Inference. Statistical
        Science, 4(3): 188 - 234, 1989.
    """

    def __init__(self, space):
        super().__init__(
            space,
            diffeo=SimplexToPositiveHypersphere(),
            image_space=Hypersphere(dim=space.dim).equip_with_metric(
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
