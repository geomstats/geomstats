import pytest

from geomstats.test.geometry.base import LevelSetTestCase
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

        for datum in vec_data:
            self.test_to_grassmannian(**datum)


class StiefelStaticMethodsTestCase(TestCase):
    def test_to_grassmannian(self, point, expected, atol):
        projected = self.Space.to_grassmannian(point)
        self.assertAllClose(projected, expected, atol=atol)
