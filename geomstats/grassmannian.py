"""
Grassmannian manifold Gr(n, p),
a set of all p-dimensional subspaces in n-dimensional space,
where p <= n
"""

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.euclidean_space import EuclideanMetric
from geomstats.matrices_space import MatricesSpace
from geomstats.riemannian_metric import RiemannianMetric

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

        raise ValueError()
        dimension = int(p * n - (p * (p + 1) / 2) - p * p)
        super(Grassmannian, self).__init__(
              dimension=dimension,
              embedding_manifold=MatricesSpace(n, p))


class GrassmannianCanonicalMetric(RiemannianMetric):

    def __init__(self, n, p):
        assert isinstance(n, int) and isinstance(p, int)
        assert p <= n
        self.n = n
        self.p = p

        dimension = int(p * n - (p * (p + 1) / 2) - p * p)
        super(GrassmannianCanonicalMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))
        self.embedding_metric = EuclideanMetric(n*p)
