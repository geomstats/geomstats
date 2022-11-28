import pytest

from geomstats.test.geometry.base import VectorSpaceTestCase
from geomstats.test.vectorization import generate_vectorization_data


class EuclideanTestCase(VectorSpaceTestCase):
    def test_exp(self, tangent_vec, base_point, expected, atol):
        exp = self.space.exp(tangent_vec, base_point)
        self.assertAllClose(exp, expected, atol=atol)

    @pytest.mark.random
    def test_exp_random(self, n_points, atol):
        tangent_vec = self.space.random_point(n_points)
        base_point = self.space.random_point(n_points)

        expected = tangent_vec + base_point

        self.test_exp(tangent_vec, base_point, expected, atol)

    @pytest.mark.vec
    def test_exp_vec(self, n_reps, atol):
        tangent_vec = self.space.random_point()
        point = self.space.random_point()

        res = self.space.exp(tangent_vec, point)

        vec_data = generate_vectorization_data(
            data=[
                dict(tangent_vec=tangent_vec, base_point=point, expected=res, atol=atol)
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="sym",
        )
        for datum in vec_data:
            self.test_exp(**datum)
