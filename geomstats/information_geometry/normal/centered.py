"""Information manifold of centered multivariate normal distributions.

Lead authors: Antoine Collas, Alice Le Brigant.
"""

from scipy.stats import multivariate_normal

import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDMatrices
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyMultivariateRandomVariable,
)
from geomstats.vectorization import broadcast_to_multibatch, get_batch_shape


class CenteredNormalDistributions(InformationManifoldMixin, SPDMatrices):
    r"""Class for the manifold of centered multivariate normal distributions.

    This is the class for multivariate normal distributions with zero mean.
    Each distribution is represented by its covariance matrix, i.e. a symmetric
    positive-definite matrix of size :math:`sample\_dim`.

    Parameters
    ----------
    sample_dim : int
        Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, sample_dim, equip=True):
        super().__init__(n=sample_dim, support_shape=(sample_dim,), equip=equip)
        self.sample_dim = sample_dim
        self._scp_rv = SharedMeanNormalDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return CenteredNormalMetric

    def sample(self, point, n_samples=1):
        """Sample from a centered multivariate normal distribution.

        Parameters
        ----------
        point : array-like, shape=[..., sample_dim, sample_dim]
            Symmetric positive definite matrix representing the covariance matrix
            of a multivariate normal distribution with zero mean.
        n_samples : int
            Number of points to sample with each covariance matrix in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, sample_dim]
            Sample from centered multivariate normal distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Parameters
        ----------
        point : array-like, shape=[..., sample_dim, sample_dim]
            Symmetric positive definite matrix representing the covariance matrix
            of a multivariate normal distribution with zero mean.

        Returns
        -------
        pdf : function
            Probability density function of the centered multivariate normal
            distributions with covariance matrices provided by point.
        """
        batch_shape = get_batch_shape(self.point_ndim, point)
        det_cov = gs.linalg.det(point)
        inv_cov = gs.linalg.inv(point)
        pdf_normalization = 1 / gs.sqrt(gs.power(2 * gs.pi, self.sample_dim) * det_cov)

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_samples, sample_dim]
                Points at which to compute the probability
                density function.

            Returns
            -------
            pdf_at_x: array-like, shape=[..., n_samples]
                Probability density function at x.
            """
            x_, inv_cov_ = broadcast_to_multibatch(
                (x.shape[0],), batch_shape, x, inv_cov
            )

            aux = gs.exp(-0.5 * gs.dot(x_, gs.matvec(inv_cov_, x_)))
            return scalarvecmul(pdf_normalization, aux)

        return pdf


class CenteredNormalMetric:
    """Class for the Fisher information metric of centered normal distributions."""

    def __new__(cls, space):
        """Instantiate a scaled SPD affine metric."""
        space.equip_with_metric(SPDAffineMetric)
        return ScalarProductMetric(space, 1 / 2)


class SharedMeanNormalDistributionsRandomVariable(ScipyMultivariateRandomVariable):
    """A multivariate normal random variable with zero mean."""

    def __init__(self, space):
        rvs = lambda point, *args, **kwargs: multivariate_normal.rvs(
            *args, cov=point, **kwargs
        )
        pdf = lambda x, point: multivariate_normal.pdf(x, cov=point)
        super().__init__(space, rvs, pdf)
