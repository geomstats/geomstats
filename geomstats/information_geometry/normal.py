"""Information Manifold of multivariate normal distributions with the Fisher metric.

Lead authors: Antoine Collas, Alice Le Brigant.
"""

import math

from scipy.stats import multivariate_normal, norm

import geomstats
import geomstats.backend as gs
import geomstats.errors as errors
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.geometry.product_manifold import ProductManifold
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDMatrices
from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.fisher_rao_metric import FisherRaoMetric


class NormalDistributions:
    """Class for the normal distributions.

    This class is a common interface to the following different situations:

    - univariate normal distributions
    - centered multivariate normal distributions
    - multivariate normal distributions with diagonal covariance matrix
    - general multivariate normal distributions

    Parameters
    ----------
    sample_dim : int
        Dimension of the sample space of the normal distribution.
    distribution_type : str, {'centered', 'diagonal', 'general'}
        Type of distributions.
        Optional, default: 'general'.
    """

    def __new__(cls, sample_dim, distribution_type="general", equip=True):
        """Instantiate class that corresponds to the distribution_type."""
        errors.check_parameter_accepted_values(
            distribution_type,
            "distribution_type",
            ["centered", "diagonal", "general"],
        )
        if sample_dim == 1:
            return UnivariateNormalDistributions(equip=equip)
        if distribution_type == "centered":
            return CenteredNormalDistributions(sample_dim, equip=equip)
        if distribution_type == "diagonal":
            return DiagonalNormalDistributions(sample_dim, equip=equip)
        return GeneralNormalDistributions(sample_dim, equip=equip)


class UnivariateNormalDistributions(InformationManifoldMixin, PoincareHalfSpace):
    """Class for the manifold of univariate normal distributions.

    This is upper half-plane.
    """

    def __init__(self, equip=True):
        super().__init__(dim=2, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return UnivariateNormalMetric

    def random_point(self, n_samples=1, bound=1.0):
        """Sample parameters of normal distributions.

        The uniform distribution on [-bound/2, bound/2]x[0, bound] is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of the square where the normal parameters are sampled.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., 2]
            Sample of points representing normal distributions.
        """
        means = -bound / 2 + bound * gs.random.rand(n_samples)
        stds = bound * gs.random.rand(n_samples)
        return gs.squeeze(gs.transpose(gs.vstack((means, stds))), axis=0)

    def sample(self, point, n_samples=1):
        """Sample from the normal distribution.

        Sample from the normal distribution with parameters provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from normal distributions.
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        samples = gs.vstack([norm.rvs(mean, scale, size=n_samples) for mean, scale in point])
        return gs.squeeze(samples, axis=0)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the normal
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        pdf : function
            Probability density function of the normal distribution with
            parameters provided by point.
        """
        mean = point[..., :1]
        std = point[..., 1:]
        pdf_normalization = 1 / gs.sqrt(2 * gs.pi * std**2)
        
        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_samples,]
                Points at which to compute the probability density function.

            Returns
            -------
            pdf_at_x : array-like, shape=[..., n_samples]
                Values of pdf at x for each value of the parameters provided
                by point.
            """
            pdf = gs.exp(-0.5 * (x - mean) ** 2 / std**2)
            return pdf_normalization * pdf

        return pdf


class CenteredNormalDistributions(InformationManifoldMixin, SPDMatrices):
    """Class for the manifold of centered multivariate normal distributions.

    This is the class for multivariate normal distributions with zero mean.
    Each distribution is represented by its covariance matrix, i.e. a symmetric
    positive-definite matrix of size :math:`sample_dim`.

    Parameters
    ----------
    sample_dim : int
        Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, sample_dim, equip=True):
        super().__init__(n=sample_dim, equip=equip)
        self.sample_dim = sample_dim

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
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, sample_dim]
            Sample from centered multivariate normal distributions.
        """
        point = gs.to_ndarray(point, to_ndim=3, axis=0)
        mean = gs.zeros(self.sample_dim)
        samples = gs.vstack([multivariate_normal.rvs(mean, cov, size=n_samples) for cov in point])
        return gs.squeeze(samples, axis=0)

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
            pdf = gs.exp(-0.5 * gs.einsum('ni,...ii,ni->n...', x, inv_cov, x))
            return gs.transpose(pdf_normalization * pdf)

        return pdf


class DiagonalNormalDistributions(InformationManifoldMixin, OpenSet):
    """Class for the manifold of diagonal multivariate normal distributions.

    This is the class for multivariate normal distributions with diagonal
    covariance matrices. Each distribution is represented by a vector of size
    :math:`2 * sample_dim` where the first :math:`sample_dim` elements contain
    the mean vector and the :math:`sample_dim` last elements contain the
    diagonal of the covariance matrix.

    Parameters
    ----------
    sample_dim : int
          Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, sample_dim, equip=True):
        self.sample_dim = sample_dim
        self.sample_space = Euclidean(dim=sample_dim, equip=False)
        dim = 2 * sample_dim
        super().__init__(dim=dim, embedding_space=Euclidean(dim), equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return DiagonalNormalMetric

    def _unstack_mean_diagonal(self, point):
        """Extract mean and diagonal of the covariance matrix from a given point.

        Parameters
        ----------
        sample_dim : int
            Dimension of the sample space of the multivariate normal distribution.
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

    def _stack_mean_diagonal(self, mean, diagonal):
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
        return gs.hstack((mean, diagonal))

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point to test. First :math:`sample_dim` elements contain the
            mean vector and the last :math:`sample_dim` elements contain
            the diagonal of the covariance matrix.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        _, diagonal = self._unstack_mean_diagonal(point)
        return gs.logical_and(point.shape[-1] == self.dim, gs.all(diagonal >= atol, axis=-1))

    def random_point(self, n_samples=1):
        """Generate random parameters of multivariate diagonal normal distributions.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., 2 * sample_dim]
            Sample of points representing multivariate diagonal normal distributions.
            First :math:`sample_dim` elements contain the mean vector and the last
            :math:`sample_dim` elements contain the diagonal of the covariance matrix.
        """
        mean = self.sample_space.random_point(n_samples=n_samples)
        diagonal_shape = (self.sample_dim,) if n_samples == 1 else (n_samples, self.sample_dim)
        diagonal = gs.reshape(gs.random.rand(n_samples * self.sample_dim) ** 2, diagonal_shape)
        return self._stack_mean_diagonal(mean, diagonal)

    def projection(self, point):
        """Project a point on the manifold of diagonal multivariate normal distribution.

        Floor the eigenvalues of the diagonal covariance matrix to gs.atol.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point to project. First :math:`sample_dim` elements contain
            the mean vector and the last :math:`sample_dim` elements contain
            the diagonal of the covariance matrix.

        Returns
        -------
        projected: array-like, shape=[..., 2 * sample_dim]
            Point containing means and diagonals
            of covariance matrices.
        """
        mean, diagonal = self._unstack_mean_diagonal(point)
        regularized = gs.where(diagonal < gs.atol, gs.atol, diagonal)
        projected = self._stack_mean_diagonal(mean, regularized)
        return projected

    def sample(self, point, n_samples=1):
        """Sample from the diagonal multivariate normal distribution.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point on the manifold. First :math:`sample_dim` elements contain
            the mean vector and the last :math:`sample_dim` elements contain
            the diagonal of the covariance matrix.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, sample_dim]
            Sample from multivariate normal distributions.
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        means, diagonals = self._unstack_mean_diagonal(point)
        covs = gs.vec_to_diag(diagonals)
        samples = gs.vstack([multivariate_normal.rvs(mean, cov, size=n_samples) for mean, cov in zip(means, covs)])
        return gs.squeeze(samples, axis=0)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Parameters
        ----------
        point : array-like, shape=[..., 2 * sample_dim]
            Point representing a probability distribution.
            First :math:`sample_dim` elements contain the mean vector and the last
            :math:`sample_dim` elements contain the diagonal of the covariance matrix.

        Returns
        -------
        pdf : function
            Probability density function of the normal distribution with
            parameters provided by point.
        """
        mean, diagonal = self._unstack_mean_diagonal(point)
        det_cov = gs.prod(diagonal, axis=-1)
        pdf_normalization = 1 / gs.sqrt(gs.power((2 * gs.pi), self.sample_dim) * det_cov)
        
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
            pdf = gs.stack([gs.exp(-0.5 * gs.sum((x_ - mean) ** 2 / diagonal, axis=-1)) for x_ in x], axis=0)
            return gs.transpose(pdf_normalization * pdf)

        return pdf


class GeneralNormalDistributions(InformationManifoldMixin, ProductManifold):
    """Class for the manifold of multivariate normal distributions.

    This is the class for multivariate normal distributions on the Euclidean space.
    Each distribution is represented by the concatenation of its mean vector and
    its covariance matrix reshaped in a vector of size :math:`sample_dim ** 2`.

    Parameters
    ----------
    sample_dim : int
        Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, sample_dim, equip=True):
        super().__init__(
            factors=(Euclidean(sample_dim), SPDMatrices(sample_dim)),
            default_point_type="vector",
            equip=equip,
        )
        self.sample_dim = sample_dim

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
        diagonal : array-like, shape=[..., sample_dim, sample_dim]
            Covariance matrices from the input point.
        """
        mean = point[..., : self.sample_dim]
        cov = point[..., self.sample_dim :]
        cov = cov.reshape((-1, self.sample_dim, self.sample_dim))
        return mean, gs.squeeze(cov, axis=0)

    def sample(self, point, n_samples=1):
        """Sample from a multivariate normal distribution.

        Parameters
        ----------
        point : array-like, shape=[..., sample_dim + sample_dim ** 2]
            Point representing a multivariate normal distribution.
            First :math:`sample_dim` elements contain the mean vector and the last
            :math:`sample_dim ** 2` elements contain the covariance matrix row by row.
        n_samples : int
            Number of points to sample with each parameter in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, sample_dim]
            Sample from multivariate normal distributions.
        """
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        means, covs = self._unstack_mean_covariance(point)
        samples = gs.vstack([multivariate_normal.rvs(mean, cov, size=n_samples) for mean, cov in zip(means, covs)])
        return gs.squeeze(samples, axis=0)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Parameters
        ----------
        point : array-like, shape=[..., sample_dim + sample_dim ** 2]
            Point representing a multivariate normal distribution.
            First :math:`sample_dim` elements contain the mean vector and the last
            :math:`sample_dim ** 2` elements contain the covariance matrix row by row.

        Returns
        -------
        pdf : function
            Probability density function of the multivariate normal
            distributions with parameters provided by point.
        """

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

            Returns
            -------
            pdf_at_x: array-like, shape=[..., n_samples]
                Probability density function at x.
            """
            print('x',x)
            print('mean', mean, 'inv_cov', inv_cov)
            pdf = gs.stack([gs.exp(-0.5 * gs.einsum('...i,...ii,...i->...', x_-mean, inv_cov, x_-mean)) for x_ in x], axis=0)
            return gs.transpose(pdf_normalization * pdf)

        return pdf


class UnivariateNormalMetric(PullbackDiffeoMetric):
    """Class for the Fisher information metric on univariate normal distributions.

    This is the pullback of the metric of the Poincare upper half-plane
    by the diffeomorphism :math:`(mean, std) -> (mean, sqrt{2} std)`.
    """

    def _define_embedding_space(self):
        r"""Define the equipped space with the metric to pull back.

        This is the metric of the Poincare upper half-plane
        with a scaling factor of 2.

        Returns
        -------
        embedding_metric : RiemannianMetric object
            The metric of the Poincare upper half-plane.
        """
        space = PoincareHalfSpace(dim=2)
        space.metric = ScalarProductMetric(PoincareHalfSpaceMetric(space), 2.0)
        return space

    def diffeomorphism(self, base_point):
        r"""Image of base point in the Poincare upper half-plane.

        This is the image by the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution. Coordinates
            are mean and standard deviation.

        Returns
        -------
        image_point : array-like, shape=[..., 2]
            Image of base_point in the Poincare upper half-plane.
        """
        return gs.stack(
            [base_point[..., 0] / gs.sqrt(2.0), base_point[..., 1]], axis=-1
        )

    def inverse_diffeomorphism(self, image_point):
        r"""Inverse image of a point in the Poincare upper half-plane.

        This is the inverse image by the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        image_point : array-like, shape=[..., 2]
            Point in the upper half-plane.

        Returns
        -------
        base_point : array-like, shape=[..., 2]
            Inverse image of the image point, representing a normal
            distribution. Coordinates are mean and standard deviation.
        """
        return gs.stack(
            [image_point[..., 0] * gs.sqrt(2.0), image_point[..., 1]], axis=-1
        )

    def tangent_diffeomorphism(self, tangent_vec, base_point):
        r"""Image of tangent vector.

        This is the image by the tangent map of the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2]
            Tangent vector at base point.

        base_point : array-like, shape=[..., 2]
            Base point representing a normal distribution.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 2]
            Image tangent vector at image of the base point.
        """
        return self.diffeomorphism(tangent_vec)

    def inverse_tangent_diffeomorphism(self, image_tangent_vec, image_point):
        r"""Inverse image of tangent vector.

        This is the inverse image by the tangent map of the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 2]
            Image of a tangent vector at image_point.

        image_point : array-like, shape=[..., 2]
            Image of a point representing a normal distribution.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 2]
            Inverse image of image_tangent_vec.
        """
        return self.inverse_diffeomorphism(image_tangent_vec)

    def metric_matrix(self, base_point):
        """Compute the metric matrix at the tangent space at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Metric matrix.
        """
        stds = base_point[..., 1]
        const = 1 / stds**2
        mat = gs.array([[1.0, 0], [0, 2]])
        return gs.einsum("...,ij->...ij", const, mat)

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Compute the sectional curvature.

        In the literature sectional curvature is noted K.

        For two orthonormal tangent vectors :math:`x,y` at a base point,
        the sectional curvature is defined by :math:`K(x,y) = <R(x, y)x, y>`.

        For non-orthonormal vectors, it is
        :math:`K(x,y) = <R(x, y)y, x> / (<x, x><y, y> - <x, y>^2)`.

        sectional_curvature(X, Y, P) = K(X,Y) where X, Y are tangent vectors
        at base point P.

        The information manifold of univariate normal distributions has constant
        sectional curvature given by :math:`K = - 1/2`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., 2]
            Point in the manifold.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.
        """
        sectional_curv = -0.5
        if (
            tangent_vec_a.ndim == 1
            and tangent_vec_b.ndim == 1
            and (base_point is None or base_point.ndim == 1)
        ):
            return gs.array(sectional_curv)

        n_sec_curv = []
        if base_point is not None and base_point.ndim == 2:
            n_sec_curv.append(base_point.shape[0])
        if tangent_vec_a.ndim == 2:
            n_sec_curv.append(tangent_vec_a.shape[0])
        if tangent_vec_b.ndim == 2:
            n_sec_curv.append(tangent_vec_b.shape[0])
        n_sec_curv = max(n_sec_curv)

        return gs.tile(sectional_curv, (n_sec_curv,))


class CenteredNormalMetric:
    """Class for the Fisher information metric of centered normal distributions."""

    def __new__(cls, space):
        """Instantiate a scaled SPD affine metric."""
        return ScalarProductMetric(SPDAffineMetric(space), 1 / 2)


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
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        tangent_vec_a = self._stacked_mean_diagonal_to_1d_pairs(tangent_vec_a)
        print('tanvec a', tangent_vec_a)
        tangent_vec_b = self._stacked_mean_diagonal_to_1d_pairs(tangent_vec_b)
        print('tanvec b', tangent_vec_b)
        base_point = self._stacked_mean_diagonal_to_1d_pairs(
            base_point, apply_sqrt=True
        )
        print('base point', base_point)
        inner_prod = self._univariate_normal.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        print('inner prod', inner_prod)
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

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2 * sample_dim]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        return math.inf
    
class GeneralNormalMetric(FisherRaoMetric):
    # def inner_product(self, base_point):
    #     pass

    def exp(self, tangent_vec, base_point):
        pass

    def log(self, point, base_point):
        pass

    def geodesic(self, initial_point, end_point=None, initial_tangent_vec=None):
        pass