import geomstats.backend as gs
from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class ProductManifoldTestCase(ManifoldTestCase):
    def test_embed_to_product_after_project_from_product(self, point, atol=gs.atol):
        """Check embedding the factors of a point recovers it.

        Parameters
        ----------
        point : array-like, shape=[..., *space.shape]
            Point on the product manifold.
        atol : float
            Absolute tolerance.
        """
        factor_point = self.space.project_from_product(point)
        point_ = self.space.embed_to_product(factor_point)

        self.assertAllClose(point_, point, atol=atol)


class ProductRiemannianMetricTestCase(RiemannianMetricTestCase):
    def test_metric_matrix_is_block_diagonal(self, base_point, atol=gs.atol):
        """Check metric matrix has the metric matrix of each factor as a block.

        Parameters
        ----------
        base_point : array-like, shape=[..., *space.shape]
            Point on the product manifold.
        atol : float
            Absolute tolerance.
        """
        res = self.space.metric.metric_matrix(base_point)
        factor_base_points = self.space.project_from_product(base_point)

        start = 0
        for factor, factor_base_point in zip(self.space.factors, factor_base_points):
            factor_matrix = factor.metric.metric_matrix(factor_base_point)
            end = start + factor_matrix.shape[-1]

            self.assertAllClose(
                res[..., start:end, start:end], factor_matrix, atol=atol
            )
            self.assertAllClose(
                res[..., start:end, :start],
                gs.zeros_like(res[..., start:end, :start]),
                atol=atol,
            )
            self.assertAllClose(
                res[..., start:end, end:],
                gs.zeros_like(res[..., start:end, end:]),
                atol=atol,
            )
            start = end
