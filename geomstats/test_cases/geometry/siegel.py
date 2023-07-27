import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data
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

    @pytest.mark.vec
    def test_tangent_vec_from_base_point_to_zero_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.tangent_vec_from_base_point_to_zero(
            tangent_vec, base_point
        )

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            n_reps=n_reps,
            vectorization_type="sym" if self.tangent_to_multiple else "repetition-0",
            expected_name="expected",
        )
        self._test_vectorization(vec_data)

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

    # @pytest.mark.vec
    # def test_sectional_curvature_at_zero_vec(self, n_reps, atol):
    #     base_point = None
    #     tangent_vec_a = self.data_generator.random_tangent_vec(base_point)
    #     tangent_vec_b = self.data_generator.random_tangent_vec(base_point)

    #     expected = self.space.metric.sectional_curvature(
    #         tangent_vec_a, tangent_vec_b
    #     )

    #     vec_data = generate_vectorization_data(
    #         data=[
    #             dict(
    #                 tangent_vec_a=tangent_vec_a,
    #                 tangent_vec_b=tangent_vec_b,
    #                 expected=expected,
    #                 atol=atol,
    #             )
    #         ],
    #         arg_names=["tangent_vec_a", "tangent_vec_b"],
    #         expected_name="expected",
    #         vectorization_type="sym" if self.tangent_to_multiple else "repeat-0-1",
    #         n_reps=n_reps,
    #     )
    #     self._test_vectorization(vec_data)
