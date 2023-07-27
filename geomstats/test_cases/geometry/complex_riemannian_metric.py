import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase


class ComplexRiemannianMetricTestCase(RiemannianMetricTestCase):
    @pytest.mark.random
    def test_inner_product_is_symmetric(self, n_points, atol):
        """Check that the inner product is Hermitian."""
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        inner_product_ab = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        inner_product_ba = self.space.metric.inner_product(
            tangent_vec_b, tangent_vec_a, base_point
        )
        self.assertAllClose(inner_product_ab, gs.conj(inner_product_ba), atol=atol)

    @pytest.mark.random
    @pytest.mark.type
    def test_inner_product_is_complex(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
        tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

        result = self.space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )
        self.assertTrue(gs.is_complex(result))

    @pytest.mark.random
    @pytest.mark.type
    def test_dist_is_real(self, n_points, atol):
        point_a = self.data_generator.random_point(n_points)
        point_b = self.data_generator.random_point(n_points)

        result = self.space.metric.dist(point_a, point_b)
        self.assertTrue(not gs.is_complex(result))

    @pytest.mark.random
    @pytest.mark.type
    def test_log_is_complex(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        point = self.data_generator.random_point(n_points)

        result = self.space.metric.log(point, base_point)
        self.assertTrue(gs.is_complex(result))

    @pytest.mark.random
    @pytest.mark.type
    def test_exp_is_complex(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        result = self.space.metric.exp(tangent_vec, base_point)
        self.assertTrue(gs.is_complex(result))
