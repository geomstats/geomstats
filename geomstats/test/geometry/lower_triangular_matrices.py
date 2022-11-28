import pytest

from geomstats.test.geometry.base import VectorSpaceTestCase
from geomstats.test.vectorization import generate_vectorization_data


class LowerTriangularMatricesTestCase(VectorSpaceTestCase):
    def test_to_vector(self, point, expected, atol):
        res = self.space.to_vector(point)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_to_vector_vec(self, n_reps, atol):
        point = self.space.random_point()
        expected = self.space.to_vector(point)

        vec_data = generate_vectorization_data(
            data=[dict(point=point, expected=expected, atol=atol)],
            arg_names=["point"],
            expected_name="expected",
            n_reps=n_reps,
        )

        for datum in vec_data:
            self.test_to_vector(**datum)
