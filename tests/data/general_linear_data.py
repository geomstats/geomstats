import random

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear, SquareMatrices
from tests.data_generation import (
    _LieGroupTestData,
    _MatrixLieAlgebraTestData,
    _OpenSetTestData,
)


class GeneralLinearTestData(_LieGroupTestData, _OpenSetTestData):
    n_list = random.sample(range(2, 5), 2)
    positive_det_list = [True, False]
    space_args_list = list(zip(n_list, positive_det_list))
    shape_list = [(n, n) for n in n_list]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = GeneralLinear

    def belongs_test_data(self):
        smoke_data = [
            dict(n=3, point=gs.eye(3), expected=True),
            dict(n=3, point=gs.ones((3, 3)), expected=False),
            dict(n=3, point=gs.ones(3), expected=False),
            dict(n=3, point=[gs.eye(3), gs.ones((3, 3))], expected=[True, False]),
        ]
        return self.generate_tests(smoke_data)

    def compose_test_data(self):
        smoke_data = [
            dict(
                n=2,
                mat1=[[1.0, 0.0], [0.0, 2.0]],
                mat2=[[2.0, 0.0], [0.0, 1.0]],
                expected=2.0 * self.Space(2).identity,
            )
        ]
        return self.generate_tests(smoke_data)

    def inv_test_data(self):
        mat_a = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        smoke_data = [
            dict(
                n=3,
                mat=gs.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]),
                expected=(
                    1.0
                    / 3.0
                    * gs.array(
                        [[-2.0, -4.0, 3.0], [-2.0, 11.0, -6.0], [3.0, -6.0, 3.0]]
                    )
                ),
            ),
            dict(
                n=3,
                mat=gs.array([mat_a, -gs.eye(3, 3)]),
                expected=gs.array([mat_a, -gs.eye(3, 3)]),
            ),
        ]
        return self.generate_tests(smoke_data)

    def exp_test_data(self):
        smoke_data = [
            dict(
                n=3,
                tangent_vec=[
                    [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                    [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                ],
                base_point=None,
                expected=[
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
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def log_test_data(self):
        smoke_data = [
            dict(
                n=3,
                point=[
                    [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
                    [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
                ],
                base_point=None,
                expected=[
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
                ],
            )
        ]
        return self.generate_tests(smoke_data)

    def orbit_test_data(self):
        point = gs.array([[gs.exp(4.0), 0.0], [0.0, gs.exp(2.0)]])
        sqrt = gs.array([[gs.exp(2.0), 0.0], [0.0, gs.exp(1.0)]])
        identity = self.Space(2).identity
        time = gs.linspace(0.0, 1.0, 3)
        smoke_data = [
            dict(
                n=2,
                point=point,
                base_point=identity,
                time=time,
                expected=gs.array([identity, sqrt, point]),
            ),
            dict(
                n=2,
                point=[point, point],
                base_point=identity,
                time=time,
                expected=[
                    gs.array([identity, sqrt, point]),
                    gs.array([identity, sqrt, point]),
                ],
            ),
        ]
        return self.generate_tests(smoke_data)

    def log_after_exp_test_data(self):
        return super().log_after_exp_test_data(amplitude=10.0)


class SquareMatricesTestData(_MatrixLieAlgebraTestData):
    n_list = random.sample(range(2, 5), 2)
    space_args_list = [(n,) for n in n_list]
    shape_list = [(n, n) for n in n_list]
    n_samples_list = random.sample(range(2, 5), 2)
    n_points_list = random.sample(range(2, 5), 2)
    n_vecs_list = random.sample(range(2, 5), 2)

    Space = SquareMatrices

    def belongs_test_data(self):
        smoke_data = [
            dict(n=3, mat=gs.eye(3), expected=True),
            dict(n=3, mat=gs.ones((3, 3)), expected=True),
            dict(n=3, mat=gs.ones(3), expected=False),
        ]
        return self.generate_tests(smoke_data)
