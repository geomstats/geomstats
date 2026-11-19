"""Information manifold of general multivariate normal distributions.

Lead authors: Antoine Collas, Alice Le Brigant.
"""

from scipy.stats import multivariate_normal

import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyMultivariateRandomVariable,
)
from geomstats.vectorization import broadcast_to_multibatch, get_batch_shape


class GeneralNormalDistributions(InformationManifoldMixin, ProductManifold):
    r"""Class for the manifold of multivariate normal distributions.

    This is the class for multivariate normal distributions on the Euclidean space.
    Each distribution is represented by the concatenation of its mean vector and
    its covariance matrix reshaped in a vector of size :math:`sample\_dim ^ 2`.

    Parameters
    ----------
    sample_dim : int
        Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, sample_dim, equip=True):
        super().__init__(
            factors=(Euclidean(sample_dim), SPDMatrices(sample_dim)),
            point_ndim=1,
            support_shape=(sample_dim,),
            equip=equip,
        )
        self.sample_dim = sample_dim
        self._scp_rv = MultivariateNormalDistributionsRandomVariable(self)

    def _unstack_mean_covariance(self, point):
        """Extract mean and covariance matrix from a given point.

        Parameters
        ----------
        point : array-like, shape=[..., sample_dim + sample_dim ** 2]
            Input point from which means and covariance matrices are extracted.

        Returns
        -------
        mean : array-like, shape=[..., sample_dim]
            Means from the input point.
        cov : array-like, shape=[..., sample_dim, sample_dim]
            Covariance matrices from the input point.
        """
        batch_shape = get_batch_shape(self.point_ndim, point)
        mean = point[..., : self.sample_dim]
        cov = point[..., self.sample_dim :]
        cov = cov.reshape(batch_shape + (self.sample_dim, self.sample_dim))
        return mean, cov

    def sample(self, point, n_samples=1):
        r"""Sample from a multivariate normal distribution.

        Parameters
        ----------
        point : array-like, shape=[..., sample_dim + sample_dim ** 2]
            Point representing a multivariate normal distribution.
            First :math:`sample\_dim` elements contain the mean vector and the last
            :math:`sample\_dim ^ 2` elements contain the covariance matrix row by row.
        n_samples : int
            Number of points to sample with each parameter in point.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, sample_dim]
            Sample from multivariate normal distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        r"""Compute pdf associated to point.

        Parameters
        ----------
        point : array-like, shape=[..., sample_dim + sample_dim ** 2]
            Point representing a multivariate normal distribution.
            First :math:`sample\_dim` elements contain the mean vector and the last
            :math:`sample\_dim ^ 2` elements contain the covariance matrix row by row.

        Returns
        -------
        pdf : function
            Probability density function of the multivariate normal
            distributions with parameters provided by point.
        """
        batch_shape = get_batch_shape(self.point_ndim, point)
        mean, cov = self._unstack_mean_covariance(point)
        det_cov = gs.linalg.det(cov)
        inv_cov = gs.linalg.inv(cov)
        pdf_normalization = 1 / gs.sqrt(gs.power(2 * gs.pi, self.sample_dim) * det_cov)

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_samples, sample_dim]
                Points at which to compute the probability
                density function.
            """
            x_, (mean_, inv_cov_) = broadcast_to_multibatch(
                (x.shape[0],), batch_shape, x, mean, inv_cov
            )
            aux_0 = x_ - mean_
            aux_1 = gs.exp(-0.5 * gs.dot(aux_0, gs.matvec(inv_cov_, aux_0)))
            return scalarvecmul(pdf_normalization, aux_1)

        return pdf


class MultivariateNormalDistributionsRandomVariable(ScipyMultivariateRandomVariable):
    """A multivariate normal random variable."""

    def __init__(self, space):
        rvs = lambda point, size: multivariate_normal.rvs(
            *space._unstack_mean_covariance(point), size=size
        )
        pdf = lambda x, point: multivariate_normal.pdf(
            x, *space._unstack_mean_covariance(point)
        )
        super().__init__(space, rvs, pdf)
