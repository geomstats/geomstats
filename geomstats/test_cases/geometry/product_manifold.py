import pytest

from geomstats.test_cases.geometry.manifold import ManifoldTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class ProductManifoldTestCase(ManifoldTestCase):
    @pytest.mark.random
    def test_embed_to_product_after_project_from_product(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        factor_point = self.space.project_from_product(point)
        point_ = self.space.embed_to_product(factor_point)

        self.assertAllClose(point_, point, atol=atol)


class ProductRiemannianMetricTestCase(RiemannianMetricTestCase):
    pass
