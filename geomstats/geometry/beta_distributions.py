"""Statistical Manifold of beta distributions with the Fisher metric."""

from scipy.special import polygamma
from scipy.stats import beta

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric


class BetaDistributions(EmbeddedManifold):
    """
    Class for the manifold of beta distributions
    """

    def __init__(self):
        super(BetaDistributions, self).__init__(
            dimension=2,
            embedding_manifold=Euclidean(dimension=2))

    def belongs(self, point, point_type=None):
        """Evaluate if a point belongs to the manifold of beta distributions.

        The statistical manifold of beta distributions is the upper right
        quadrant of the euclidean 2-plane.

        Parameters
        ----------
        point
        point_type

        Returns
        -------

        """
        point = gs.to_ndarray(point, to_ndim=2)
        n_points, point_dim = point.shape
        belongs = point_dim == self.dimension
        belongs = gs.to_ndarray(belongs, to_ndim=1)
        belongs = gs.tile(belongs, n_points)
        belongs = belongs * gs.greater(point, 0).all(axis=1)
        if n_points == 1:
            belongs = belongs[0]
        return belongs

    def random_uniform(self, n_samples=1, bound=10.0):
        """Sample parameters of beta distributions with the uniform
        distribution.

        Parameters
        ----------
        n_samples : int, optional
        bound: float, optional

        Returns
        -------
        samples : array-like, shape=[n_samples, 2]
        """
        return bound * gs.random.rand(n_samples, 2)

    def maximum_likelihood_fit(self, data, loc=0, scale=0):
        parameters = []
        for sample in data:
            a, b, _, _ = beta.fit(sample, floc=loc, fscale=scale)
            parameters.append(gs.array([a, b]))
        return gs.stack(parameters)


class BetaMetric(RiemannianMetric):

    def __init__(self):
        super(RiemannianMetric, self).__init__(dimension=2)

    @staticmethod
    def detg(a, b):
        detg = polygamma(1, a) * polygamma(1, b) - polygamma(1, a + b) * (
                    polygamma(1, a) + polygamma(1, b))
        return detg

    def inner_product_matrix(self, base_point=None):
        """
        Inner product matrix at the tangent space at a base point.
        Parameters
        ----------
        base_point : array-like, shape=[n_samples, dimension], optional
        """
        assert ~ (base_point is None), 'The metric depends on the base point'
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        matrices = []
        for point in base_point:
            param_a, param_b = point
            g0 = gs.array([polygamma(1, param_a) - polygamma(1, param_a +
                                                             param_b),
                           - polygamma(1, param_a + param_b)])
            g1 = gs.array([- polygamma(1, param_a + param_b),
                           polygamma(1, param_b) -
                           polygamma(1, param_a + param_b)])
            matrices.append(gs.stack([g0, g1]))
        return gs.stack(matrices)

    def christoffels(self, base_point):
        """Compute Christoffel symbols.

        Compute the Christoffel symbols of the Fisher metric on Beta
        distributions.
        """

        def coefficients(param_a, param_b):
            detg = self.detg(param_a, param_b)
            c1 = (polygamma(2, param_a) * polygamma(1, param_b) -
                  polygamma(2, param_a) * polygamma(1, param_a + param_b) -
                  polygamma(1, param_b) * polygamma(2, param_a + param_b)) / \
                 (2 * detg)
            c2 = - polygamma(1, param_b) * polygamma(2, param_a + param_b) / (
                    2 * detg)
            c3 = (polygamma(2, param_b) * polygamma(1, param_a + param_b) -
                  polygamma(1, param_b) * polygamma(2, param_a + param_b)) / (
                    2 * detg)
            return c1, c2, c3

        assert ~ (base_point is None), 'The Christoffels require a base point'
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        param_a, param_b = base_point[:, 0], base_point[:, 1]
        c1, c2, c3 = coefficients(param_a, param_b)
        c4, c5, c6 = coefficients(param_b, param_a)
        christoffel = []
        for d1, d2, d3, d4, d5, d6 in zip(c1, c2, c3, c4, c5, c6):
            gamma_0 = gs.array([[d1, d2], [d2, d3]])
            gamma_1 = gs.array([[d6, d5], [d5, d4]])
            christoffel.append(gs.stack([gamma_0, gamma_1]))
        return gs.stack(christoffel)
