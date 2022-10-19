"""Statistical Manifold of multivaraite normal distributions with the Fisher metric.

Lead author: XXXX.
"""

from scipy.stats import norm, multivariate_normal

import geomstats
import geomstats.backend as gs
from geomstats.geometry.base import OpenSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.information_geometry.base import InformationManifoldMixin


class MultivariateDiagonalNormalDistributions(OpenSet, InformationManifoldMixin):
    """Class for the manifold of diagonal multivariate normal distributions."""

    def __init__(self, n):
        self.n = n
        self.euclidean_n = Euclidean(dim=n)
        dim = int(2 * n)
        super().__init__(
            dim=dim,
            embedding_space=Euclidean(dim)
        )

    def _get_location_and_diagonal(self, point):
        """Extract location and diagonal of the covariance matrix
        from a given point/tangent vector.

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
        location = point[..., :self.n]
        diagonal = point[..., self.n:]
        return location, diagonal

    def _set_location_and_diagonal(self, location, diagonal):
        """Set location and diagonal of the covariance matrix
        into a point.

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

    def belongs(self, point, atol=gs.atol):
        """Evaluate if the point belongs to the
        multivariate diagonal Normal distribution manifold.

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
        _, diagonal = self._get_location_and_diagonal(point)
        belongs = gs.logical_and(belongs, gs.all(diagonal >= atol, axis=-1))
        return belongs

    def random_point(self, n_samples=1):
        """Sample parameters of normal distributions.

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
        location = self.euclidean_n.random_point(
            n_samples=n_samples, bound=bound)
        if n_samples == 1:
            diagonal = gs.array(norm.rvs(size=(n,))**2)
        else:
            diagonal = gs.array(norm.rvs(size=(n_samples, n))**2)
        point = self._set_location_and_diagonal(location, diagonal)
        return point

    def projection(self, point):
        """Project a 2*n vector on the manifold of
        diagonal mutlivariate normal distributions.

        The eigenvalues are floored to gs.atol.

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
        location, diagonal = self._get_location_and_diagonal(point)
        regularized = gs.where(diagonal < gs.atol, gs.atol, diagonal)
        projected = self._set_location_and_diagonal(location, regularized)
        return projected

    def sample(self, point, n_samples=1):
        """Sample from the diagonal mutltivariate normal distribution.

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
            loc, diag = self._get_location_and_diagonal(p)
            cov = gs.vec_to_diag(diag)
            samples.append(
                gs.array(multivariate_normal.rvs(loc, cov, size=n_samples))
            )
        return samples[0] if point.shape[0] == 1 else gs.stack(samples)

    @staticmethod
    def pdf(x, point):
        """Generate parameterized function for pdf.

        Parameters
        ----------
        x : array-like, shape=[n_points, dim]
            Points at which to compute the probability
            density function.
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.

        Returns
        -------
        pdf_at_x : array-like, shape=[..., n_points]
            Values of pdf at x for each value of the parameters provided
            by point.
        """
        raise NotImplementedError("The pdf method has not yet been implemented.")

    @staticmethod
    def cdf(x, point):
        """Generate parameterized function for cdf.

        Parameters
        ----------
        x : array-like, shape=[n_points, dim]
            Points at which to compute the probability
            density function.
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.

        Returns
        -------
        cdf_at_x : array-like, shape=[..., n_points]
            Values of cdf at x for each value of the parameters provided
            by point.
        """
        raise NotImplementedError("The cdf method has not yet been implemented.")
