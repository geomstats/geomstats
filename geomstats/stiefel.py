"""
Stiefel manifold St(n,p),
a set of all orthonormal p-frames in n-dimensional space,
where p <= n
"""

import geomstats.backend as gs

from geomstats.embedded_manifold import EmbeddedManifold
from geomstats.matrices_space import MatricesSpace

TOLERANCE = 1e-6
EPSILON = 1e-6


class Stiefel(EmbeddedManifold):
    """
    Class for Stiefel manifolds St(n,p),
    a set of all orthonormal p-frames in n-dimensional space,
    where p <= n.
    """
    def __init__(self, n, p):
        assert isinstance(n, int) and isinstance(p, int)
        assert p <= n

        self.n = n
        self.p = p

        dimension = int(p * n - (p * (p + 1) / 2))
        super(Stiefel, self).__init__(
              dimension=dimension,
              embedding_manifold=MatricesSpace(n, p))

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Evaluate if a point belongs to St(n,p),
        i.e. if it is a p-frame in n-dimensional space,
        and it is orthonormal.
        """
        point = gs.to_ndarray(point, to_ndim=3)
        n_points, n, p = point.shape

        if (n, p) != (self.n, self.p):
            return gs.array([[False]] * n_points)

        point_transpose = gs.transpose(point, axes=(0, 2, 1))
        diff = gs.matmul(point_transpose, point) - gs.eye(p)

        diff_norm = gs.norm(diff, axis=(1, 2))
        belongs = gs.less_equal(diff_norm, tolerance)

        belongs = gs.to_ndarray(belongs, to_ndim=2)
        return belongs

    def random_uniform(self, n_samples=1):
        """
        Sample on St(n,p) with the uniform distribution.

        If Z(p,n) ~ N(0,1), then St(n,p) ~ U, according to Haar measure:
        St(n,p) := Z(Z^TZ)^{-1/2}
        """
        std_normal = gs.random.normal(size=(n_samples, self.n, self.p))
        std_normal_transpose = gs.transpose(std_normal, axes=(0, 2, 1))
        aux = gs.matmul(std_normal_transpose, std_normal)
        sqrt_aux = gs.sqrtm(aux)
        inv_sqrt_aux = gs.linalg.inv(sqrt_aux)

        return gs.matmul(std_normal, inv_sqrt_aux)
