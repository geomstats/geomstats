import random

import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import OpenSetTestData
from .lie_group import MatrixLieGroupTestData
from .matrices import MatricesMetricTestData


class GeneralLinearTestData(MatrixLieGroupTestData, OpenSetTestData):
    trials = 3
    xfails = ("exp_after_log",)

    def orbit_vec_test_data(self):
        n_times = random.sample(range(1, 5), 1)
        data = [dict(n_reps=n_reps, n_times=n_times) for n_reps in self.N_VEC_REPS]
        return self.generate_tests(data)


class GeneralLinearMatricesMetricTestData(MatricesMetricTestData):
    trials = 3
    fail_for_not_implemented_errors = False
    tolerances = {
        "exp_belongs": {"atol": 1e-4},
        "geodesic_bvp_belongs": {"atol": 1e-4},
        "geodesic_ivp_belongs": {"atol": 1e-4},
    }
    xfails = tuple(tolerances.keys())


class GeneralLinear2TestData(TestData):
    def compose_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 2.0]]),
                point_b=gs.array([[2.0, 0.0], [0.0, 1.0]]),
                expected=2.0 * gs.eye(2),
            )
        ]
        return self.generate_tests(data)

    def orbit_test_data(self):
        point = gs.array([[gs.exp(4.0), 0.0], [0.0, gs.exp(2.0)]])
        sqrt = gs.array([[gs.exp(2.0), 0.0], [0.0, gs.exp(1.0)]])
        identity = gs.eye(2)
        time = gs.linspace(0.0, 1.0, 3)
        data = [
            dict(
                point=point,
                base_point=identity,
                time=time,
                expected=gs.array([identity, sqrt, point]),
            ),
            dict(
                point=gs.stack([point, point]),
                base_point=identity,
                time=time,
                expected=gs.stack(
                    [
                        gs.array([identity, sqrt, point]),
                        gs.array([identity, sqrt, point]),
                    ]
                ),
            ),
        ]
        return self.generate_tests(data)


class GeneralLinear3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.eye(3), expected=True),
            dict(point=gs.ones((3, 3)), expected=False),
            dict(point=gs.ones(3), expected=False),
            dict(
                point=gs.stack([gs.eye(3), gs.ones((3, 3))]),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(data)

    def inverse_test_data(self):
        mat_a = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]),
                expected=(
                    1.0
                    / 3.0
                    * gs.array(
                        [[-2.0, -4.0, 3.0], [-2.0, 11.0, -6.0], [3.0, -6.0, 3.0]]
                    )
                ),
            ),
            dict(
                point=gs.array([mat_a, -gs.eye(3, 3)]),
                expected=gs.array([mat_a, -gs.eye(3, 3)]),
            ),
        ]
        return self.generate_tests(data)

    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array(
                    [
                        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                        [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    ]
                ),
                base_point=None,
                expected=gs.array(
                    [
                        [
                            [7.38905609, 0.0, 0.0],
                            [0.0, 20.0855369, 0.0],
                            [0.0, 0.0, 54.5981500],
                        ],
                        [
                            [2.718281828, 0.0, 0.0],
                            [0.0, 148.413159, 0.0],
                            [0.0, 0.0, 403.42879349],
                        ],
                    ]
                ),
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array(
                    [
                        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                        [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                    ]
                ),
                base_point=None,
                expected=gs.array(
                    [
                        [
                            [0.693147180, 0.0, 0.0],
                            [0.0, 1.09861228866, 0.0],
                            [0.0, 0.0, 1.38629436],
                        ],
                        [
                            [0.0, 0.0, 0.0],
                            [0.0, 1.609437912, 0.0],
                            [0.0, 0.0, 1.79175946],
                        ],
                    ]
                ),
            )
        ]
        return self.generate_tests(data)


class SquareMatrices3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.eye(3), expected=gs.array(True)),
            dict(point=gs.ones((3, 3)), expected=gs.array(True)),
            dict(point=gs.ones(3), expected=gs.array(False)),
        ]
        return self.generate_tests(data)
