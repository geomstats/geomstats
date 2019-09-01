"""
Grassmannian manifold Gr(n, p),
a set of all p-dimensional subspaces in n-dimensional space,
where p <= n
"""

from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean_space import EuclideanMetric
from geomstats.geometry.matrices_space import MatricesSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-5
EPSILON = 1e-6


class Grassmannian(EmbeddedManifold):
    """
    Class for Grassmannian manifolds Gr(n, p),
    a set of all p-dimensional subspaces in n-dimensional space,
    where p <= n.
    """
    def __init__(self, n, p):
        assert isinstance(n, int) and isinstance(p, int)
        assert p <= n

        self.n = n
        self.p = p

        dimension = int(p * (n - p))
        super(Grassmannian, self).__init__(
              dimension=dimension,
              embedding_manifold=MatricesSpace(n, p))


class GrassmannianCanonicalMetric(RiemannianMetric):

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
