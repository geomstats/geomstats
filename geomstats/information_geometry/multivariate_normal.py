"""Information Manifold of multivariate normal distributions with
the Fisher information metric.

Lead author: Antoine Collas.
"""

import math

from scipy.stats import multivariate_normal, norm

import geomstats
import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.information_geometry.base import InformationManifoldMixin
from geomstats.information_geometry.normal import NormalMetric


class MultivariateDiagonalNormalDistributions(InformationManifoldMixin, OpenSet):
    """Class for the manifold of diagonal multivariate normal distributions.

    This is the class for multivariate normal distributions with diagonal
    covariance matrices and samples on the $n$-dimensional Euclidean space.
    Each distribution is represented by a vector of size $2n$ where the first
    $n$ elements contain the mean vector and the $n$ last elements contain
    the diagonal of the covariance matrix.

    Parameters
    ----------
    n : int
          Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, n):
        self.n = n
        self.euclidean_n = Euclidean(dim=n)
        dim = int(2 * n)
        super().__init__(dim=dim, embedding_space=Euclidean(dim))
        self.metric = MultivariateDiagonalNormalMetric(n=n)
        self._unstack_location_diagonal = self.metric._unstack_location_diagonal
        self._stack_location_diagonal = self.metric._stack_location_diagonal

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the manifold.

        First $n$ elements contain the mean vector and the $n$ last
        elements contain the diagonal of the covariance matrix.

        Parameters
        ----------
        point : array-like, shape=[..., 2*n]
            Point to test.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the space.
        """
        point_dim = point.shape[-1]
        belongs = point_dim == self.dim
        _, diagonal = self._unstack_location_diagonal(point)
        belongs = gs.logical_and(belongs, gs.all(diagonal >= atol, axis=-1))
        return belongs

    def random_point(self, n_samples=1):
        """Generate random parameters of multivariate diagonal normal distributions.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., 2*n]
            Sample of points representing multivariate diagonal
            normal distributions.
        """
        n = self.n
        bound = 1.0
        location = self.euclidean_n.random_point(n_samples=n_samples, bound=bound)
        if n_samples == 1:
            diagonal = gs.array(norm.rvs(size=(n,)) ** 2)
        else:
            diagonal = gs.array(norm.rvs(size=(n_samples, n)) ** 2)
        point = self._stack_location_diagonal(location, diagonal)
        return point

    def projection(self, point):
        """Project a point on the manifold of
        diagonal multivariate normal distributions.

        Floor the eigenvalues of the diagonal covariance matrix to gs.atol.

        Parameters
        ----------
        point : array-like, shape=[..., 2*n]
            Point to project.

        Returns
        -------
        projected: array-like, shape=[..., 2*n]
            Point containing locations and diagonals
            of covariance matrices.
        """
        location, diagonal = self._unstack_location_diagonal(point)
        regularized = gs.where(diagonal < gs.atol, gs.atol, diagonal)
        projected = self._stack_location_diagonal(location, regularized)
        return projected

    def sample(self, point, n_samples=1):
        """Sample from the diagonal multivariate normal distribution.

        Parameters
        ----------
        point : array-like, shape=[..., 2*n]
            Point on the manifold.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples, n]
            Sample from multivariate normal distributions.
        """
        geomstats.errors.check_belongs(point, self)
        if point.ndim > 2:
            raise NotImplementedError
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for p in point:
            loc, diag = self._unstack_location_diagonal(p)
            cov = gs.vec_to_diag(diag)
            samples.append(gs.array(multivariate_normal.rvs(loc, cov, size=n_samples)))
        return samples[0] if point.shape[0] == 1 else gs.stack(samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Parameters
        ----------
        point : array-like, shape=[..., 2*n]
            Point representing a probability distribution.

        Returns
        -------
        pdf : function
            Probability density function of the normal distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)
        if point.ndim > 2:
            raise NotImplementedError
        point = gs.to_ndarray(point, to_ndim=2, axis=0)
        point = point[:, None, :]
        location, diagonal = self._unstack_location_diagonal(point)
        n = self.n

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_samples, n]
                Points at which to compute the probability
                density function.
            """
            x = gs.to_ndarray(x, to_ndim=2, axis=0)
            x = x[None, :, :]
            det_cov = gs.squeeze(gs.prod(diagonal, axis=-1))
            tmp_0 = 1 / gs.sqrt(gs.power((2 * gs.pi), n) * det_cov)
            tmp_1 = gs.exp(-0.5 * gs.sum(((x - location) ** 2) / diagonal, axis=-1))
            while tmp_0.ndim < tmp_1.ndim:
                tmp_0 = tmp_0[..., None]
            pdf_at_x = tmp_0 * tmp_1
            pdf_at_x = gs.squeeze(pdf_at_x)
            return pdf_at_x

        return pdf


class MultivariateDiagonalNormalMetric(RiemannianMetric):
    """Class for the Fisher information metric of diagonal normal distributions.

    Parameters
    ----------
    n : int
          Dimension of the sample space of the multivariate normal distribution.
    """

    def __init__(self, n):
        self.n = n
        dim = int(2 * n)
        super().__init__(dim=dim)
        self.univariate_normal_metric = NormalMetric()

    def _unstack_location_diagonal(self, point):
        """Extract location and diagonal of the covariance matrix from a given point.

        Parameters
        ----------
        point : array-like, shape=[..., 2*n]
            Input point from which locations and diagonals are extracted.

        Returns
        -------
        location : array-like, shape=[..., n]
            Locations from the input point.
        diagonal : array-like, shape=[..., n]
            Diagonals of covariance matrices from the input point.
        """
        location = point[..., : self.n]
        diagonal = point[..., self.n :]
        return location, diagonal

    def _stack_location_diagonal(self, location, diagonal):
        """Set location and diagonal of the covariance matrix into a point.

        Parameters
        ----------
        location : array-like, shape=[..., n]
            Locations to stack.
        diagonal : array-like, shape=[..., n]
            Diagonals of covariance matrices from the input point.

        Returns
        -------
        point : array-like, shape=[..., 2*n]
            Point with locations and diagonals covariance matrices.
        """
        point = gs.concatenate([location, diagonal], axis=-1)
        return point

    def _stacked_location_diagonal_to_1d_pairs(self, point, apply_sqrt=False):
        """Create pairs of 1d parameters from nd counterparts.

        Parameters
        ----------
        point: array-like, shape=[..., 2*n]
            Stacked point (e.g. stacked locations and diagonals).
        apply_sqrt: bool
            Determine if a square root is applied to the diagonals.

        Returns
        -------
        pairs : array-like, shape=[..., n, 2]
            Pairs of parameters (e.g. locations and variances).
        """
        location, diagonal = self._unstack_location_diagonal(point)
        if apply_sqrt:
            diagonal = gs.sqrt(diagonal)
        point = gs.stack([location, diagonal], axis=-1)
        return point

    def _1d_pairs_to_stacked_location_diagonal(self, point, apply_square=False):
        """Create nd stacked parameters from pairs of 1d counterparts.

        Parameters
        ----------
        pairs : array-like, shape=[..., n, 2]
            Pairs of parameters (e.g. locations and variances).
        apply_square: bool
            Determine if a square is applied to the diagonals.

        Returns
        -------
        point: array-like, shape=[..., 2*n]
            Stacked point (e.g. stacked locations and diagonals).
        """
        location = point[..., 0]
        diagonal = point[..., 1]
        if apply_square:
            diagonal = gs.power(diagonal, 2)
        point = gs.concatenate([location, diagonal], axis=-1)
        return point

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a: array-like, shape=[..., 2*n]
            Tangent vector at base point.
        tangent_vec_b: array-like, shape=[..., 2*n]
            Tangent vector at base point.
        base_point: array-like, shape=[..., 2*n]
            Base point.
            Optional, default: None.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        tangent_vec_a = self._stacked_location_diagonal_to_1d_pairs(tangent_vec_a)
        tangent_vec_b = self._stacked_location_diagonal_to_1d_pairs(tangent_vec_b)
        base_point = self._stacked_location_diagonal_to_1d_pairs(
            base_point, apply_sqrt=True
        )
        inner_prod = self.univariate_normal_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        inner_prod = gs.sum(inner_prod, axis=-1)
        return inner_prod

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2*n]
            Tangent vector at the base point.
        base_point : array-like, shape=[..., 2*n]
            Point.

        Returns
        -------
        end_point : array-like, shape=[..., 2*n]
            Point reached by the geodesic starting from `base_point`
            with initial velocity `tangent_vec`
        """
        tangent_vec = self._stacked_location_diagonal_to_1d_pairs(tangent_vec)
        base_point = self._stacked_location_diagonal_to_1d_pairs(
            base_point, apply_sqrt=True
        )
        end_point = self.univariate_normal_metric.exp(tangent_vec, base_point)
        end_point = self._1d_pairs_to_stacked_location_diagonal(
            end_point, apply_square=True
        )
        return end_point

    def log(self, point, base_point):
        """Compute Riemannian logarithm of a point wrt a base point.

        Parameters
        ----------
        point : array-like, shape=[..., 2*n]
            Point.
        base_point : array-like, shape=[..., 2*n]
            Point.

        Returns
        -------
        log : array-like, shape=[..., 2*n]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        point = self._stacked_location_diagonal_to_1d_pairs(point, apply_sqrt=True)
        base_point = self._stacked_location_diagonal_to_1d_pairs(
            base_point, apply_sqrt=True
        )
        log = self.univariate_normal_metric.log(point, base_point)
        log = self._1d_pairs_to_stacked_location_diagonal(log)
        return log

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2*n]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        return math.inf
