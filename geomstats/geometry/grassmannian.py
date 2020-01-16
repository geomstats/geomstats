"""
Grassmannian manifold Gr(n, p),
a set of all p-dimensional subspaces in n-dimensional space,
where p <= n
"""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean_space import EuclideanMetric
from geomstats.geometry.matrices_space import MatricesSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-5
EPSILON = 1e-6


class Grassmannian(EmbeddedManifold):
    """
    Class for Grassmann manifolds Gr(n, k)
    of k-dimensional subspaces in the n-dimensional euclidean space.
    """
    def __init__(self, n, k):
        assert isinstance(n, int) and isinstance(k, int)
        assert k <= n

        self.n = n
        self.k = k

        dimension = int(k * (n - k))
        super(Grassmannian, self).__init__(
              dimension=dimension,
              embedding_manifold=MatricesSpace(n, n))

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Check if an (n,n)-matrix is an orthogonal projector
        onto a subspace of rank p.
        """
        raise NotImplementedError(
                'The Grassmann `belongs` is not implemented.'
                'It shall test whether p*=p, p^2 = p and rank(p) = k.')


class GrassmannianCanonicalMetric(RiemannianMetric):
    """
    Canonical metric of the Grassmann manifold.

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

    def exp(self, vector, point):
        """
        Exponentiate the invariant vector field v from base point p.

        `vector` is skew-symmetric, in so(n).
        `point` is a rank p projector of Gr(n, k).

        Parameters
        ----------
        vector : array-like, shape=[n_samples, n, n]
        point : array-like, shape=[n_samples, n, n]

        Returns
        -------
        exp : array-like, shape=[n_samples, n, n]
        """
        expm = gs.linalg.expm
        mul = MatricesSpace.mul
        return mul(mul(expm(vector), point), expm(-vector))
