import math

import pytest

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.test.test_case import TestCase


class AlgebraUtilsTestCase(TestCase):
    @pytest.mark.random
    def test_taylor_functions_even_exp(self, taylor_function, exponent, atol):
        x = 10 ** (-exponent)
        expected = taylor_function["function"](math.sqrt(x))
        result = utils.taylor_exp_even_func(x, taylor_function, order=4)
        self.assertAllClose(result, expected, atol=atol)

    def test_from_vector_to_diagonal_matrix(
        self,
        vector,
        expected,
        atol,
        num_diag=0,
    ):
        res = utils.from_vector_to_diagonal_matrix(vector, num_diag=num_diag)
        self.assertAllClose(res, expected, atol=atol)

    @pytest.mark.random
    def test_rotate_points(self, n_points, atol):
        north_pole = gs.array([1.0, 0.0, 0.0])
        sphere = Hypersphere(dim=2, equip=False)

        end_point = sphere.random_point()
        res = utils.rotate_points(north_pole, end_point)
        self.assertAllClose(res, end_point, atol=atol)

        points = sphere.random_point(n_points)
        res = utils.rotate_points(points, north_pole)
        self.assertAllClose(res, points, atol=atol)
