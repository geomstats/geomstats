"""Statistical Manifold of normal distributions with the Fisher metric."""

from scipy.stats import norm

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.poincare_half_space import PoincareHalfSpaceMetric


class NormalDistributions(PoincareHalfSpace):
    """Class for the manifold of univariate normal distributions.

    This is upper half-plane.
    """

    def __init__(self):
        super(NormalDistributions, self).__init__(dim=2)
        self.metric = FisherRaoMetric()

    @staticmethod
    def random_point(n_samples=1, bound=1.):
        """Sample parameters of normal distributions.

        The uniform distribution on [-bound/2, bound/2]x[0, bound] is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of the square where the normal parameters are sampled.
            Optional, default: 5.

        Returns
        -------
        samples : array-like, shape=[..., 2]
            Sample of points representing normal distributions.
        """
        means = -bound + 2 * bound * gs.random.rand(n_samples)
        stds = bound * gs.random.rand(n_samples)
        if n_samples == 1:
            return gs.array((means[0], stds[0]))
        return gs.transpose(gs.stack((means, stds)))

    def sample(self, point, n_samples=1):
        """Sample from the normal distribution.

        Sample from the normal distribution with parameters provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a normal distribution (location and scale).
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from normal distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for loc, scale in point:
            samples.append(gs.array(
                norm.rvs(loc, scale, size=n_samples)))
        return samples[0] if len(point) == 1 else gs.stack(samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the normal
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a normal distribution (location and scale).

        Returns
        -------
        pdf : function
            Probability density function of the normal distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)
        means = point[..., 0]
        stds = point[..., 1]

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points,]
                Points at which to compute the probability density function.
            """
            x = gs.array(x, gs.float32)
            x = gs.to_ndarray(x, to_ndim=1)

            pdf_at_x = [
                gs.array(norm.pdf(x, loc=mean, scale=std)) for mean, std
                in zip(means, stds)]
            pdf_at_x = gs.stack(pdf_at_x, axis=-1)

            return pdf_at_x
        return pdf


class FisherRaoMetric(PoincareHalfSpaceMetric):
    """Class for the Fisher information metric on normal distributions.

    This is the metric of the Poincare upper half-plane.
    """

    def __init__(self):
        super(FisherRaoMetric, self).__init__(dim=2)
