import pytest

from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.riemannian_metric import RiemannianMetricTestCase
from geomstats.test_cases.information_geometry.base import (
    InformationManifoldMixinTestCase,
)


class GammaDistributionsTestCase(InformationManifoldMixinTestCase, OpenSetTestCase):
    def test_natural_to_standard(self, point, expected, atol):
        res = self.space.natural_to_standard(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_natural_to_standard_vec(self, n_reps, atol):
        point = self.data_generator.random_point()

        expected = self.space.natural_to_standard(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_standard_to_natural(self, point, expected, atol):
        res = self.space.standard_to_natural(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_standard_to_natural_vec(self, n_reps, atol):
        point = self.data_generator.random_point_standard()

        expected = self.space.standard_to_natural(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_standard_to_natural_after_natural_to_standard(self, n_points, atol):
        point = self.data_generator.random_point(n_points)

        point_standard = self.space.natural_to_standard(point)
        point_ = self.space.standard_to_natural(point_standard)

        self.assertAllClose(point_, point, atol=atol)

    def test_tangent_natural_to_standard(self, vec, base_point, expected, atol):
        res = self.space.tangent_natural_to_standard(vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_natural_to_standard_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.tangent_natural_to_standard(vec, base_point)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, base_point=base_point, expected=expected, atol=atol)],
            arg_names=["vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_tangent_standard_to_natural(self, vec, base_point, expected, atol):
        res = self.space.tangent_standard_to_natural(vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_tangent_standard_to_natural_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point_standard()
        vec = self.data_generator.random_tangent_vec_standard(base_point)

        expected = self.space.tangent_standard_to_natural(vec, base_point)

        vec_data = generate_vectorization_data(
            data=[dict(vec=vec, base_point=base_point, expected=expected, atol=atol)],
            arg_names=["vec", "base_point"],
            expected_name="expected",
            vectorization_type="sym" if self.tangent_to_multiple else "repeat-0",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_tangent_standard_to_natural_after_tangent_natural_to_standard(
        self, n_points, atol
    ):
        base_point = self.data_generator.random_point(n_points)
        vec = self.data_generator.random_tangent_vec(base_point)

        vec_standard = self.space.tangent_natural_to_standard(vec, base_point)
        base_point_standard = self.space.natural_to_standard(base_point)

        vec_ = self.space.tangent_standard_to_natural(vec_standard, base_point_standard)
        self.assertAllClose(vec_, vec, atol=atol)


class GammaMetricTestCase(RiemannianMetricTestCase):
    def test_jacobian_christoffels(self, base_point, expected, atol):
        res = self.space.metric.jacobian_christoffels(base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_jacobian_christoffels_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()

        expected = self.space.metric.jacobian_christoffels(base_point)

        vec_data = generate_vectorization_data(
            data=[dict(base_point=base_point, expected=expected, atol=atol)],
            arg_names=["base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
