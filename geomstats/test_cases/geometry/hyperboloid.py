import geomstats.backend as gs
from geomstats.geometry.minkowski import Minkowski
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class HyperboloidMetricTestCase(RiemannianMetricTestCase):
    def test_inner_product_is_minkowski_inner_product(
        self, tangent_vec_a, tangent_vec_b, base_point, atol=gs.atol
    ):
        """Check inner product is the ambient Minkowski inner product.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hyperboloid.
        atol : float
            Absolute tolerance.
        """
        minkowski = Minkowski(self.space.dim + 1)

        res = self.space.metric.inner_product(tangent_vec_a, tangent_vec_b, base_point)
        expected = minkowski.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertAllClose(res, expected, atol=atol)
