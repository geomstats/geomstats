import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

from scipy.stats import binom


class BinomialDistributions(OpenSet):
    """Class for the manifold of binomial distributions."""

    def __init__(self, n_draws):
        super(BinomialDistributions, self).__init__(dim=1, ambient_space=Euclidean(dim=1), metric = BinomialFisherRaoMetric(n_draws))
        self.n_draws = n_draws

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the manifold of exponential distributions.

        Parameters
        ----------
        point : array-like, shape=[...]
            Point to be checked.
        atol : float
            Tolerance to evaluate positivity.
            Optional, default: gs.atol

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean indicating whether point represents a binomial
            distribution.
        """
        belongs = len(point.shape) == 1
        belongs = gs.logical_and(belongs, gs.logical_and(atol <= point, point <= 1-atol))
        return belongs

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
        samples : array-like, shape=[..., 1]
            Sample of points representing binomial distributions.
        """
        return gs.random.rand(n_samples)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The parameter is floored to `gs.atol` if it is negative, and to '1-gs.atol' if it is greater than 1.

        Parameters
        ----------
        point : array-like, shape=[...]
            Point in ambient space.
        atol : float
            Tolerance to evaluate positivity.

        Returns
        -------
        projected : array-like, shape=[...]
            Projected point.
        """
        return gs.where(gs.logical_or(point < atol, point > 1-atol), int(point > 1-atol) - atol, point)

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
        samples = []
        for i in range(n_samples):
            samples.append(binom.rvs(self.n_draws, point))
        return samples[0] if len(point) == 1 else gs.transpose(gs.stack(samples))

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the exponential
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing an exponential distribution (location and scale).

        Returns
        -------
        pdf : function
            Probability density function of the exponential distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)

        def pdf(k):
            """Generate parameterized function for exponential pdf.

            Parameters
            ----------
            k : array-like, shape=[n_points,]
                Integers in \{0, ..., n_draws\} at which to compute the probability mass function.
            """
            k = gs.array(k, gs.float32)
            k = gs.to_ndarray(k, to_ndim=1)

            pdf_at_k = [gs.array(binom.pmf(k, self.n_draws, param)) for param in list(point)]
            pdf_at_k = gs.stack(pdf_at_k, axis=-1)

            return pdf_at_k

        return pdf


class BinomialFisherRaoMetric(RiemannianMetric):
    """Class for the Fisher information metric on exponential distributions."""

    def __init__(self, n_draws):
        super(BinomialFisherRaoMetric, self).__init__(dim=1)
        self.n_draws = n_draws

    def squared_dist(self, point_a, point_b, **kwargs):
        return 4 * self.n_draws * (gs.arcsin(gs.sqrt(point_a))-gs.arcsin(gs.sqrt(point_b)))**2