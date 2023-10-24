import math
import random

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.test.data import TestData


class AlgebraUtilsTestData(TestData):
    def taylor_functions_even_exp_test_data(self):
        functions = [
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

        exponents = [random.randint(4, 12) for _ in range(2)]

        data = []
        for function in functions:
            for exponent in exponents:
                data.append(dict(taylor_function=function, exponent=exponent))

        return self.generate_tests(data)

    def from_vector_to_diagonal_matrix_test_data(self):
        data = [
            dict(
                vector=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                num_diag=0,
                expected=gs.array(
                    [
                        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
                        [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    ]
                ),
            ),
            dict(
                vector=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                num_diag=1,
                expected=gs.array(
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
                ),
            ),
            dict(
                vector=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                num_diag=-1,
                expected=gs.array(
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
                ),
            ),
        ]
        return self.generate_tests(data)

    def rotate_points_test_data(self):
        return self.generate_random_data()
