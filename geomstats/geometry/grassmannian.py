"""Module exposing `Grassmannian` and `GrassmannianMetric` classes."""

import geomstats.backend as gs
import geomstats.error
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
    subspaces in the n-dimensional euclidean space.
    """

    def __init__(self, n, k):
        geomstats.error.check_integer(k, 'k')
        geomstats.error.check_integer(n, 'n')
        if k > n:
            raise ValueError(
                'k <= n is required: k-dimensional subspaces in n dimensions.')

        dimension = int(k * (n - k))
        super(Grassmannian, self).__init__(
            dimension=dimension,
            embedding_manifold=Matrices(n, n))

        self.n = n
        self.k = k
        self.metric = GrassmannianCanonicalMetric(3, 2)

    def belongs(self, point, tolerance=TOLERANCE):
        """Check if the point belongs to the manifold.

        Check if an (n,n)-matrix is an orthogonal projector
        onto a subspace of rank p.

        Parameters
        ----------
        point
        tolerance

        Returns
        -------
        belongs : bool
        """
        raise NotImplementedError(
            'The Grassmann `belongs` is not implemented.'
            'It shall test whether p*=p, p^2 = p and rank(p) = k.')


class GrassmannianCanonicalMetric(RiemannianMetric):
    """Canonical metric of the Grassmann manifold.

    Coincides with the Frobenius metric.
    """

    def __init__(self, n, p):
        geomstats.error.check_integer(p, 'p')
        geomstats.error.check_integer(n, 'n')
        if p > n:
            raise ValueError('p <= n is required.')

        dimension = int(p * (n - p))
        super(GrassmannianCanonicalMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))

        self.n = n
        self.p = p
        self.embedding_metric = EuclideanMetric(n * p)

    def exp(self, tangent_vec, base_point):
        """Exponentiate the invariant vector field v from base point p.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, n, n]
            `vector` is skew-symmetric, in so(n).
        point : array-like, shape=[n_samples, n, n]
            `point` is a rank p projector of Gr(n, k).

        Returns
        -------
        exp : array-like, shape=[n_samples, n, n]
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
        point : array-like, shape=[n_samples, n, n]
            Point in the Grassmannian.
        base_point : array-like, shape=[n_samples, n, n]
            Point in the Grassmannian.

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, n, n]
            Tangent vector at `base_point`.

        References
        ----------
        .. [Batzies2015] Batzies, Hüper, Machado, Leite.
            "Geometric Mean and Geodesic Regression on Grassmannians"
            Linear Algebra and its Applications, 466, 83-101, 2015.
        """
        GLn = GeneralLinear(self.n)
        id_n = GLn.identity()
        sym2 = 2 * point - id_n
        sym1 = 2 * base_point - id_n
        rot = GLn.mul(sym2, sym1)
        return GLn.log(rot) / 2
