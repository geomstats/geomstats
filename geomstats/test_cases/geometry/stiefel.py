import pytest

from geomstats.test.geometry.base import LevelSetTestCase, RiemannianMetricTestCase
from geomstats.test.test_case import TestCase
from geomstats.test.vectorization import generate_vectorization_data


class StiefelTestCase(LevelSetTestCase):
    def test_to_grassmannian(self, point, expected, atol):
        projected = self.space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)

    @pytest.mark.vec
    def test_to_grassmannian_vec(self, n_reps, atol):
        point = self.space.random_point()
        expected = self.space.to_grassmannian(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)


class StiefelStaticMethodsTestCase(TestCase):
    def test_to_grassmannian(self, point, expected, atol):
        projected = self.Space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)


class StiefelCanonicalMetricTestCase(RiemannianMetricTestCase):
    def test_retraction(self, tangent_vec, base_point, expected, atol):
        res = self.space.metric.retraction(tangent_vec, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_retraction_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = self.space.metric.retraction(tangent_vec, base_point)

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
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)

    def test_lifting(self, point, base_point, expected, atol):
        res = self.space.metric.lifting(point, base_point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_lifting_vec(self, n_reps, atol):
        point = self.data_generator.random_point()
        base_point = self.data_generator.random_point()

        expected = self.space.metric.lifting(point, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(point=point, base_point=base_point, expected=expected, atol=atol)
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
