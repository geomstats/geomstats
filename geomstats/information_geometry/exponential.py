import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric

from scipy.stats import expon


class ExponentialDistributions(OpenSet):
    """Class for the manifold of exponential distributions."""

    def __init__(self):
        super(ExponentialDistributions, self).__init__(dim=1, ambient_space=Euclidean(dim=1), metric=ExponentialFisherRaoMetric())
        self.metric = ExponentialFisherRaoMetric()

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
            Boolean indicating whether point represents an exponential
            distribution.
        """
        belongs = len(point.shape) == 1
        belongs = gs.logical_and(belongs, point >= atol)
        return belongs

    @staticmethod
    def random_point(n_samples=1, bound=1.0):
        """Sample parameters of exponential distributions.

        The uniform distribution on [0, bound] is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Right-end ot the segment where exponential parameters are sampled.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[...]
            Sample of points representing exponential distributions.
        """
        return bound * gs.random.rand(n_samples)

    def projection(self, point, atol=gs.atol):
        """Project a point in ambient space to the open set.

        The last coordinate is floored to `gs.atol` if it is negative.

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
        return gs.where(point < atol, atol, point)

    def sample(self, point, n_samples=1):
        """Sample from the exponential distribution.

        Sample from the exponential distribution with parameter provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[...]
            Point representing an exponential distribution.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from exponential distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=1)
        samples = []
        for i in range(n_samples):
            samples.append(expon.rvs(scale=point))
        return samples[0] if len(point) == 1 else gs.transpose(gs.stack(samples))

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the exponential
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[...]
            Point representing an exponential distribution (scale).

        Returns
        -------
        pdf : function
            Probability density function of the exponential distribution with
            scale parameter provided by point.
        """
        geomstats.errors.check_belongs(point, self)

        def pdf(x):
            """Generate parameterized function for exponential pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points,]
                Points at which to compute the probability density function.
            """
            x = gs.array(x, gs.float32)
            x = gs.to_ndarray(x, to_ndim=1)

            pdf_at_x = [gs.array(expon.pdf(x, loc=0, scale=param)) for param in list(point)]
            pdf_at_x = gs.stack(pdf_at_x, axis=-1)

            return pdf_at_x

        return pdf


class ExponentialFisherRaoMetric(RiemannianMetric):
    """Class for the Fisher information metric on exponential distributions."""

    def __init__(self):
        super(ExponentialFisherRaoMetric, self).__init__(dim=1)

    def squared_dist(self, point_a, point_b, **kwargs):
        return gs.abs(gs.log(point_a/point_b))
