"""Module exposing `Grassmannian` and `GrassmannianMetric` classes."""

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.general_linear import GeneralLinear
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-5
EPSILON = 1e-6


class Grassmannian(EmbeddedManifold):
    """Class for Grassmann manifolds Gr(n, k).

    Class for Grassmann manifolds Gr(n, k) of k-dimensional
    subspaces in the n-dimensional Euclidean space.

    The subspaces are represented by their (unique) orthogonal projection
    matrix onto themselves.

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    k : int
        Dimension of the subspaces.
    """

    def __init__(self, n, k):
        geomstats.errors.check_integer(k, 'k')
        geomstats.errors.check_integer(n, 'n')
        if k > n:
            raise ValueError(
                'k <= n is required: k-dimensional subspaces in n dimensions.')

        self.n = n
        self.k = k
        self.metric = GrassmannianCanonicalMetric(n, k)

        dim = int(k * (n - k))
        super(Grassmannian, self).__init__(
            dim=dim,
            embedding_manifold=Matrices(n, n),
            default_point_type='matrix')

        self.n = n
        self.k = k
        self.metric = GrassmannianCanonicalMetric(3, 2)

    def belongs(self, point, tolerance=TOLERANCE):
        """Check if the point belongs to the manifold.

        Check if an (n,n)-matrix is an orthogonal projector
        onto a subspace of rank p.

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point to be checked.
        tolerance : int
            Optional, default: 1e-5.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the Grassmannian.
        """
        point = gs.to_ndarray(point, to_ndim=3)

        if not gs.all(self._check_square(point)):
            raise ValueError('all points must be square.')

        symm = self._check_symmetric(point)
        idem = self._check_idempotent(point, tolerance)
        rank = self._check_rank(point, self.k, tolerance)

        belongs = gs.all(gs.stack([symm, idem, rank], axis=0), axis=0)

        return belongs

    @staticmethod
    def _check_square(point):
        """Check if a point is square.

        Parameters
        ----------
        point
        n: Euclidean dimension
        k: subspace dimension

        Returns
        -------
        belongs : bool
        """

        [n_points, n, p] = point.shape

        return [n == p] * n_points

    @staticmethod
    def _check_symmetric(point):
        """Check that a point is a symmetric.

        Parameters
        ----------
        point

        Returns
        -------
        belongs : bool
        """

        return Matrices.is_symmetric(point)

    @staticmethod
    def _check_idempotent(point, tolerance):
        """Check that a point is idempotent.

        Parameters
        ----------
        point
        tolerance

        Returns
        -------
        belongs : bool
        """

        diff = gs.einsum('...ij,...jk->...ik', point, point) - point
        diff_norm = gs.linalg.norm(diff, axis=(1, 2))

        return gs.less_equal(diff_norm, tolerance)

    @staticmethod
    def _check_rank(point, rank, tolerance):
        """Check that the rank of the point is equal to the
        subspace dimension.  Matrix rank is equal to number of
        singular values greater than 0.

        Parameters
        ----------
        point
        rank
        tolerance

        Returns
        -------
        belongs : bool
        """

        [_, s, _] = gs.linalg.svd(point)

        return gs.sum(s > tolerance, axis=1) == rank


class GrassmannianCanonicalMetric(RiemannianMetric):
    """Canonical metric of the Grassmann manifold.

    Coincides with the Frobenius metric.

    Parameters
    ----------
    n : int
        Dimension of the Euclidean space.
    k : int
        Dimension of the subspaces.
    """

    def __init__(self, n, p):
        geomstats.errors.check_integer(p, 'p')
        geomstats.errors.check_integer(n, 'n')
        if p > n:
            raise ValueError('p <= n is required.')

        dim = int(p * (n - p))
        super(GrassmannianCanonicalMetric, self).__init__(
            dim=dim,
            signature=(dim, 0, 0))

        self.n = n
        self.p = p
        self.embedding_metric = EuclideanMetric(n * p)

    def exp(self, tangent_vec, base_point):
        """Exponentiate the invariant vector field v from base point p.

        Parameters
        ----------
        vector : array-like, shape=[..., n, n]
            Tangent vector at base point.
            `vector` is skew-symmetric, in so(n).
        base_point : array-like, shape=[..., n, n]
            Base point.
            `base_point` is a rank p projector of Gr(n, k).

        Returns
        -------
        exp : array-like, shape=[..., n, n]
            Riemannian exponential.
        """
        expm = gs.linalg.expm
        mul = Matrices.mul
        return mul(expm(tangent_vec), base_point, expm(-tangent_vec))

    def log(self, point, base_point):
        r"""Compute the Riemannian logarithm of point w.r.t. base_point.

        Given :math:`P, P'` in Gr(n, k) the logarithm from :math:`P`
        to :math:`P` is given by the infinitesimal rotation [Batzies2015]_:

        .. math::

            \omega = \frac 1 2 \log \big((2 P' - 1)(2 P - 1)\big)

        Parameters
        ----------
        point : array-like, shape=[..., n, n]
            Point.
        base_point : array-like, shape=[..., n, n]
            Base point.

        Returns
        -------
        tangent_vec : array-like, shape=[..., n, n]
            Riemannian logarithm, a tangent vector at `base_point`.

        References
        ----------
        .. [Batzies2015] Batzies, Hüper, Machado, Leite.
            "Geometric Mean and Geodesic Regression on Grassmannians"
            Linear Algebra and its Applications, 466, 83-101, 2015.
        """
        GLn = GeneralLinear(self.n)
        id_n = GLn.identity
        sym2 = 2 * point - id_n
        sym1 = 2 * base_point - id_n
        rot = GLn.mul(sym2, sym1)
        return GLn.log(rot) / 2
