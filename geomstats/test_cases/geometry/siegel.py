import pytest

import geomstats.backend as gs
from geomstats.test_cases.geometry.complex_riemannian_metric import (
    ComplexRiemannianMetricTestCase,
)


class SiegelMetricTestCase(ComplexRiemannianMetricTestCase):
    def test_tangent_vec_from_base_point_to_zero(
        self, tangent_vec, base_point, expected, atol
    ):
        tangent_vec_at_zero = self.space.metric.tangent_vec_from_base_point_to_zero(
            tangent_vec, base_point
        )
        self.assertAllClose(tangent_vec_at_zero, expected, atol=atol)

    @pytest.mark.random
    def test_tangent_vec_from_base_point_to_zero_is_tangent(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        tangent_vec_at_zero = self.space.metric.tangent_vec_from_base_point_to_zero(
            tangent_vec, base_point
        )
        zero = gs.zeros(self.space.shape)
        is_tangent = self.space.is_tangent(tangent_vec_at_zero, zero, atol=atol)

        expected = gs.ones(n_points, dtype=bool)
        self.assertAllEqual(is_tangent, expected)

    # def test_sectional_curvature_at_zero(
    #     self, tangent_vec_a, tangent_vec_b, base_point, expected, atol
    # ):
    #     res = self.space.metric.sectional_curvature(
    #         tangent_vec_a, tangent_vec_b, base_point
    #     )
    #     self.assertAllClose(res, expected, atol=atol)
