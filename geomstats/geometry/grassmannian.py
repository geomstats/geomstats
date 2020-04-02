"""Module exposing `Grassmannian` and `GrassmannianMetric` classes."""

import geomstats.backend as gs
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
        assert isinstance(n, int) and isinstance(k, int)
        assert k <= n

        self.n = n
        self.k = k
        self.metric = GrassmannianCanonicalMetric(3, 2)

        dimension = int(k * (n - k))
        super(Grassmannian, self).__init__(
            dimension=dimension,
            embedding_manifold=Matrices(n, n))

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

    def is_tangent(self, tangent_vec, point):
        is_inf_rot = Matrices.is_skew_symmetric(tangent_vec)
        is_transverse = Matrices.equal(
                Matrices.bracket(tangent_vec, point), 
                point)
        return gs.and(is_inf_rot, is_transverse)

    def to_tangent(self, tangent_vec, point):
        inf_rot = Matrices.to_skew_symmetric(tangent_vec)
        return Matrices.bracket(inf_rot, point)

class GrassmannianCanonicalMetric(RiemannianMetric):
    """Canonical metric of the Grassmann manifold.

    Coincides with the Frobenius metric.
    """

    def __init__(self, n, p):
        assert isinstance(n, int) and isinstance(p, int)
        assert p <= n
        self.n = n
        self.p = p

        dimension = int(p * (n - p))
        super(GrassmannianCanonicalMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
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
