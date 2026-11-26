"""Information manifold of diagonal multivariate normal distributions.

Lead authors: Antoine Collas, Alice Le Brigant.
"""

import math

from scipy.stats import multivariate_normal, norm

import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.base import VectorSpaceOpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyMultivariateRandomVariable,
)
from geomstats.information_geometry.normal.univariate import (
    UnivariateNormalDistributions,
)
from geomstats.vectorization import broadcast_to_multibatch, get_batch_shape, repeat_out


class DiagonalNormalDistributions(InformationManifoldMixin, VectorSpaceOpenSet):
    r"""Class for the manifold of diagonal multivariate normal distributions.

    This is the class for multivariate normal distributions with diagonal
    covariance matrices. Each distribution is represented by a vector of size
    :math:`2 * sample\_dim` where the first :math:`sample\_dim` elements contain
    the mean vector and the :math:`sample\_dim` last elements contain the
    diagonal of the covariance matrix.

    Parameters
    ----------
    sample_dim : int
        Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, sample_dim, equip=True):
        self.sample_dim = sample_dim
        self.sample_space = Euclidean(dim=sample_dim, equip=False)
        dim = int(2 * sample_dim)
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim),
            support_shape=(sample_dim,),
            equip=equip,
        )
        self._scp_rv = DiagonalNormalDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return DiagonalNormalMetric

    def _unstack_mean_diagonal(self, point):
        """Extract mean and diagonal of the covariance matrix from a given point.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Input point from which means and diagonals are extracted.

        Returns
        -------
        mean : array-like, shape=[..., sample_dim]
            Means from the input point.
        diagonal : array-like, shape=[..., sample_dim]
            Diagonals of covariance matrices from the input point.
        """
        mean = point[..., : self.sample_dim]
        diagonal = point[..., self.sample_dim :]
        return mean, diagonal

    @staticmethod
    def _stack_mean_diagonal(mean, diagonal):
        """Set mean and diagonal of the covariance matrix into a point.

        Parameters
        ----------
        mean : array-like, shape=[..., sample_dim]
            Means to stack.
        diagonal : array-like, shape=[..., sample_dim]
            Diagonals of covariance matrices from the input point.

        Returns
        -------
        point : array-like, shape=[..., 2 * sample_dim]
            Point with means and diagonals covariance matrices.
        """
        return gs.concatenate([mean, diagonal], axis=-1)

    def belongs(self, point, atol=gs.atol):
        r"""Evaluate if the point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point to test. First :math:`sample\_dim` elements contain the
            mean vector and the last :math:`sample\_dim` elements contain
            the diagonal of the covariance matrix.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        _, diagonal = self._unstack_mean_diagonal(point)
        return gs.logical_and(belongs, gs.all(diagonal >= -atol, axis=-1))

    def random_point(self, n_samples=1):
        r"""Generate random parameters of multivariate diagonal normal distributions.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : array-like, shape=[..., 2 * sample_dim]
            Sample of points representing multivariate diagonal normal distributions.
            First :math:`sample\_dim` elements contain the mean vector and the last
            :math:`sample\_dim` elements contain the diagonal of the covariance matrix.
        """
        sample_dim = self.sample_dim
        bound = 1.0
        mean = self.sample_space.random_point(n_samples=n_samples, bound=bound)
        if n_samples == 1:
            diagonal = gs.array(norm.rvs(size=(sample_dim,)) ** 2)
        else:
            diagonal = gs.array(norm.rvs(size=(n_samples, sample_dim)) ** 2)
        point = self._stack_mean_diagonal(mean, diagonal)
        return point

    def projection(self, point):
        r"""Project a point on the manifold.

        Floor the eigenvalues of the diagonal covariance matrix to gs.atol.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point to project. First :math:`sample\_dim` elements contain
            the mean vector and the last :math:`sample\_dim` elements contain
            the diagonal of the covariance matrix.

        Returns
        -------
        projected: array-like, shape=[..., 2 * sample_dim]
            Point containing means and diagonals
            of covariance matrices.
        """
        mean, diagonal = self._unstack_mean_diagonal(point)
        regularized = gs.where(diagonal < gs.atol, gs.atol, diagonal)
        return self._stack_mean_diagonal(mean, regularized)

    def sample(self, point, n_samples=1):
        r"""Sample from the diagonal multivariate normal distribution.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point on the manifold. First :math:`sample\_dim` elements contain
            the mean vector and the last :math:`sample\_dim` elements contain
            the diagonal of the covariance matrix.
        n_samples : int
            Number of points to sample with each pair of parameters in point.

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
        point : array-like, shape=[..., 2 * sample_dim]
            Point representing a probability distribution.
            First :math:`sample\_dim` elements contain the mean vector and the last
            :math:`sample\_dim` elements contain the diagonal of the covariance matrix.

        Returns
        -------
        pdf : function
            Probability density function of the normal distribution with
            parameters provided by point.
        """
        batch_shape = get_batch_shape(self.point_ndim, point)
        n = self.sample_dim
        mean, diagonal = self._unstack_mean_diagonal(point)
        det_cov = gs.prod(diagonal, axis=-1)
        pdf_normalization = 1 / gs.sqrt(gs.power((2 * gs.pi), n) * det_cov)

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_samples, sample_dim]
                Points at which to compute the probability
                density function.
            """
            x_, (mean_, diagonal_) = broadcast_to_multibatch(
                (x.shape[0],), batch_shape, x, mean, diagonal
            )

            aux = gs.exp(-0.5 * gs.sum(((x_ - mean_) ** 2) / diagonal_, axis=-1))
            return scalarvecmul(pdf_normalization, aux)

        return pdf


class DiagonalNormalMetric(RiemannianMetric):
    """Class for the Fisher information metric of diagonal normal distributions."""

    def __init__(self, space):
        super().__init__(space=space)
        self._univariate_normal = UnivariateNormalDistributions()

    def _stacked_mean_diagonal_to_1d_pairs(self, point, apply_sqrt=False):
        """Create pairs of 1d parameters from nd counterparts.

        Parameters
        ----------
        point: array-like, shape=[..., 2 * sample_dim]
            Stacked point (e.g. stacked means and diagonals).
        apply_sqrt: bool
            Determine if a square root is applied to the diagonals.

        Returns
        -------
        pairs : array-like, shape=[..., sample_dim, 2]
            Pairs of parameters (e.g. means and variances).
        """
        mean, diagonal = self._space._unstack_mean_diagonal(point)
        if apply_sqrt:
            diagonal = gs.sqrt(diagonal)
        return gs.stack([mean, diagonal], axis=-1)

    def _1d_pairs_to_stacked_mean_diagonal(self, point, apply_square=False):
        """Create nd stacked parameters from pairs of 1d counterparts.

        Parameters
        ----------
        pairs : array-like, shape=[..., sample_dim, 2]
            Pairs of parameters (e.g. means and variances).
        apply_square: bool
            Determine if a square is applied to the diagonals.

        Returns
        -------
        point: array-like, shape=[..., 2 * sample_dim]
            Stacked point (e.g. stacked means and diagonals).
        """
        mean = point[..., 0]
        diagonal = point[..., 1]
        if apply_square:
            diagonal = gs.power(diagonal, 2)
        return gs.concatenate([mean, diagonal], axis=-1)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., 2 * sample_dim]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., 2 * sample_dim]
            Tangent vector at base point.
        base_point: array-like, shape=[..., 2 * sample_dim]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        tangent_vec_a = self._stacked_mean_diagonal_to_1d_pairs(tangent_vec_a)
        tangent_vec_b = self._stacked_mean_diagonal_to_1d_pairs(tangent_vec_b)
        base_point = self._stacked_mean_diagonal_to_1d_pairs(
            base_point, apply_sqrt=True
        )
        inner_prod = self._univariate_normal.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        return gs.sum(inner_prod, axis=-1)

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2 * sample_dim]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., 2 * sample_dim]
            Point.

        Returns
        -------
        end_point : array-like, shape=[..., 2 * sample_dim]
            Point reached by the geodesic starting from `base_point`
            with initial velocity `tangent_vec`.
        """
        tangent_vec = self._stacked_mean_diagonal_to_1d_pairs(tangent_vec)
        base_point = self._stacked_mean_diagonal_to_1d_pairs(
            base_point, apply_sqrt=True
        )
        end_point = self._univariate_normal.metric.exp(tangent_vec, base_point)
        return self._1d_pairs_to_stacked_mean_diagonal(end_point, apply_square=True)

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point.
        base_point : array-like, shape=[..., 2 * sample_dim]
            Point.

        Returns
        -------
        log : array-like, shape=[..., 2 * sample_dim]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        point = self._stacked_mean_diagonal_to_1d_pairs(point, apply_sqrt=True)
        base_point = self._stacked_mean_diagonal_to_1d_pairs(
            base_point, apply_sqrt=True
        )
        log = self._univariate_normal.metric.log(point, base_point)
        return self._1d_pairs_to_stacked_mean_diagonal(log)

    def injectivity_radius(self, base_point=None):
        """Compute the radius of the injectivity domain.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2 * sample_dim]
            Point on the manifold.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(math.inf)
        return repeat_out(self._space.point_ndim, radius, base_point)


class DiagonalNormalDistributionsRandomVariable(ScipyMultivariateRandomVariable):
    """A diagonal multivariate normal random variable."""

    # TODO: update other children
    def __init__(self, space):
        rvs = lambda point, size: multivariate_normal.rvs(
            *space._unstack_mean_diagonal(point), size=size
        )
        pdf = lambda x, point: multivariate_normal.pdf(
            x, *space._unstack_mean_diagonal(point)
        )
        super().__init__(space, rvs, pdf)
