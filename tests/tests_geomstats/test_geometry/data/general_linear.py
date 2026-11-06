from polpo.testing.data import CaseData

import geomstats.backend as gs

from .base import VectorSpaceOpenSetTestData
from .lie_group import MatrixLieGroupTestData
from .matrices import MatricesMetricTestData


class GeneralLinearTestData(MatrixLieGroupTestData, VectorSpaceOpenSetTestData):
    def __init__(self, space=None, **kwargs):
        super().__init__(space, **kwargs)

        # the matrix logarithm is not defined everywhere on GL(n)
        xfails = set(self.method_marks.get("xfail", ())) | {
            "exp_after_log",
            "log_after_exp",
        }
        self.method_marks = {**self.method_marks, "xfail": xfails}

    def orbit_vec_test_data(self):
        return self.generate_timed_vectorization_data(
            arg_names=("point", "base_point"),
            op_name="orbit",
        )


class GeneralLinearMatricesMetricTestData(MatricesMetricTestData):
    def __init__(self, space=None, **kwargs):
        super().__init__(space, **kwargs)

        # straight lines may leave GL(n)
        xfails = set(self.method_marks.get("xfail", ())) | {
            "exp_belongs",
            "geodesic_bvp_belongs",
            "geodesic_ivp_belongs",
        }
        self.method_marks = {**self.method_marks, "xfail": xfails}


class GeneralLinear2TestData(CaseData):
    def compose_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 2.0]]),
                point_b=gs.array([[2.0, 0.0], [0.0, 1.0]]),
                expected=2.0 * gs.eye(2),
            )
        ]
        return data

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
                expected=gs.stack([identity, sqrt, point]),
            ),
            dict(
                point=gs.stack([point, point]),
                base_point=identity,
                time=time,
                expected=gs.stack(
                    [
                        gs.stack([identity, sqrt, point]),
                        gs.stack([identity, sqrt, point]),
                    ]
                ),
            ),
        ]
        return data


class GeneralLinear3TestData(CaseData):
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
        return data

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
        return data

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
        return data

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
        return data


class SquareMatrices3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.eye(3), expected=gs.array(True)),
            dict(point=gs.ones((3, 3)), expected=gs.array(True)),
            dict(point=gs.ones(3), expected=gs.array(False)),
        ]
        return data
