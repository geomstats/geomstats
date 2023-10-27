import pytest

import geomstats.backend as gs
from geomstats.test.vectorization import generate_vectorization_data
from geomstats.test_cases.geometry.base import OpenSetTestCase
from geomstats.test_cases.geometry.lie_group import MatrixLieGroupTestCase


class GeneralLinearTestCase(MatrixLieGroupTestCase, OpenSetTestCase):
    def test_orbit(self, point, base_point, time, expected, atol):
        path = self.space.orbit(point, base_point)
        res = path(time)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.vec
    def test_orbit_vec(self, n_reps, n_times, atol):
        point = self.space.random_point()
        base_point = self.space.random_point()

        path = self.space.orbit(point, base_point)
        time = gs.random.uniform(size=n_times)
        expected = path(time)

        vec_data = generate_vectorization_data(
            data=[
                dict(
                    point=point,
                    base_point=base_point,
                    time=time,
                    expected=expected,
                    atol=atol,
                )
            ],
            arg_names=["point", "base_point"],
            expected_name="expected",
            n_reps=n_reps,
        )
        self._test_vectorization(vec_data)
