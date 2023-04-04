import pytest

import geomstats.backend as gs
from geomstats.test.geometry.base import VectorSpaceTestCase
from geomstats.test.vectorization import generate_vectorization_data


class EuclideanTestCase(VectorSpaceTestCase):
    def test_exp(self, tangent_vec, base_point, expected, atol):
        exp = self.space.exp(tangent_vec, base_point)
        self.assertAllClose(exp, expected, atol=atol)

    @pytest.mark.random
    def test_exp_random(self, n_points, atol):
        base_point = self.data_generator.random_point(n_points)
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        expected = tangent_vec + base_point

        self.test_exp(tangent_vec, base_point, expected, atol)

    @pytest.mark.vec
    def test_exp_vec(self, n_reps, atol):
        base_point = self.data_generator.random_point()
        tangent_vec = self.data_generator.random_tangent_vec(base_point)

        res = self.space.exp(tangent_vec, base_point)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    tangent_vec=tangent_vec,
                    base_point=base_point,
                    expected=res,
                    atol=atol,
                )
            ],
            arg_names=["tangent_vec", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
            vectorization_type="sym",
        )
        self._test_vectorization(vec_data)

    @pytest.mark.random
    def test_identity_belongs(self, atol):
        self.test_belongs(self.space.identity, gs.array(True), atol)
