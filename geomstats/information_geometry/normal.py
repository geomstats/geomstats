"""Statistical Manifold of normal distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

from scipy.stats import norm

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.information_geometry.information_manifold import InformationManifold


class NormalDistributions(PoincareHalfSpace, InformationManifold):
    """Class for the manifold of univariate normal distributions.

    This is upper half-plane.
    """

    def __init__(self):
        super().__init__(dim=2)
        self.shape = (2,)
        self.metric = NormalMetric()

    @staticmethod
    def random_point(n_samples=1, bound=1.0):
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
            samples.append(gs.array(norm.rvs(loc, scale, size=n_samples)))
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
        means = gs.to_ndarray(means, to_ndim=1)
        stds = gs.to_ndarray(stds, to_ndim=1)

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points,]
                Points at which to compute the probability density function.
            """
            pdf_at_x = [
                gs.array(
                    (
                        (1.0 / gs.sqrt(2 * gs.pi * std**2))
                        * gs.exp(-((x - mean) ** 2) / (2 * std**2))
                    )
                )
                for mean, std in zip(means, stds)
            ]
            pdf_at_x = gs.stack(pdf_at_x, axis=-1)

            return pdf_at_x

        return pdf


class NormalMetric(PoincareHalfSpaceMetric):
    """Class for the Fisher information metric on normal distributions.

    This is the metric of the Poincare upper half-plane.
    """

    def __init__(self):
        super().__init__(dim=2)

    def metric_matrix(self, base_point=None):
        """Compute the metric matrix at the tangent space at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (location and scale).

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Metric matrix.
        """
        geomstats.errors.check_belongs(base_point, self.embedding_manifold)
        stds = base_point[..., 1]
        stds = gs.to_ndarray(stds, to_ndim=1)
        mat = gs.stack(
            [gs.array(((1.0 / std**2) * gs.eye(2),)) for std in stds],
            axis=-3,
        )
        return mat
        stds = base_point[..., 1]
        stds = gs.to_ndarray(stds, to_ndim=1)
        metric_mat = gs.stack(
            [gs.array([[1.0 / std**2, 0.0], [0.0, 2.0 / std**2]]) for std in stds],
            axis=0,
        )

        if metric_mat.ndim == 3 and metric_mat.shape[0] == 1:
            return metric_mat[0]
        return metric_mat
