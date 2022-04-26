import math

import geomstats.algebra_utils as utils
import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere


class TestAlgebraUtils(geomstats.tests.TestCase):
    def setup_method(self):
        self.functions = [
            utils.cos_close_0,
            utils.sinc_close_0,
            utils.inv_sinc_close_0,
            utils.inv_tanc_close_0,
            {
                "coefficients": utils.arctanh_card_close_0["coefficients"],
                "function": lambda x: math.atanh(x) / x,
            },
            {
                "coefficients": utils.cosc_close_0["coefficients"],
                "function": lambda x: (1 - math.cos(x)) / x**2,
            },
            utils.sinch_close_0,
            utils.cosh_close_0,
            {
                "coefficients": utils.inv_sinch_close_0["coefficients"],
                "function": lambda x: x / math.sinh(x),
            },
            {
                "coefficients": utils.inv_tanh_close_0["coefficients"],
                "function": lambda x: x / math.tanh(x),
            },
        ]

    def test_all(self):
        for taylor_function in self.functions:
            for exponent in range(4, 12, 2):
                x = 10 ** (-exponent)
                expected = taylor_function["function"](math.sqrt(x))
                result = utils.taylor_exp_even_func(x, taylor_function, order=4)
                self.assertAllClose(result, expected)

    def test_rotate_points(self):
        sphere = Hypersphere(2)
        end_point = sphere.random_uniform()
        north_pole = gs.array([1.0, 0.0, 0.0])
        result = utils.rotate_points(north_pole, end_point)
        expected = end_point
        self.assertAllClose(result, expected)

        points = sphere.random_uniform(10)
        result = utils.rotate_points(points, north_pole)
        self.assertAllClose(result, points)

        points = gs.concatenate([north_pole[None, :], points])
        result = utils.rotate_points(points, end_point)
        self.assertAllClose(result[0], end_point)

    def test_from_vector_to_diagonal_matrix(self):
        vec = gs.array([1.0, 2.0, 3.0])
        mat_diag = utils.from_vector_to_diagonal_matrix(vec, -1)
        result = mat_diag.shape
        expected = (4, 4)
        self.assertAllClose(result, expected)

        vec = gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mat_diag = utils.from_vector_to_diagonal_matrix(vec, 0)
        expected = gs.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
            ]
        )
        self.assertAllClose(mat_diag, expected)

        mat_plus = utils.from_vector_to_diagonal_matrix(vec, 1)
        expected = gs.array(
            [
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 4.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0],
                    [0.0, 0.0, 0.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        self.assertAllClose(mat_plus, expected)

        mat_minus = utils.from_vector_to_diagonal_matrix(vec, -1)
        expected = gs.array(
            [
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0, 0.0],
                ],
            ]
        )
        self.assertAllClose(mat_minus, expected)
