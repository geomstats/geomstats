"""
Grassmannian manifold Gr(n, p),
a set of all p-dimensional subspaces in n-dimensional space,
where p <= n
"""

from geomstats.embedded_manifold import EmbeddedManifold
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


class GrassmannianCanonicalMetric(RiemannianMetric):

    def __init__(self, n, p):
        assert isinstance(n, int) and isinstance(p, int)
        assert p <= n
        self.n = n
        self.p = p
