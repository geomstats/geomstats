"""Unit tests for the General Linear group."""


import geomstats.backend as gs
import tests.helper as helper
from geomstats.geometry.general_linear import GeneralLinear
from tests.conftest import Parametrizer, TestCase, TestData

RTOL = 1e-5


class TestGeneralLinear(TestCase, metaclass=Parametrizer):
    cls = GeneralLinear

    class TestDataGeneralLinear(TestData):
        def belongs_data(self):
            smoke_data = [
                dict(n=3, mat=gs.eye(3), expected=True),
                dict(n=3, mat=gs.ones((3, 3)), expected=False),
                dict(n=3, mat=gs.ones(3), expected=False),
            ]
            self.generate_tests(smoke_data)

        def compose_data(self):
            smoke_data = [
                dict(
                    n=2,
                    mat1=[[1.0, 0.0], [0.0, 2.0]],
                    mat2=[[2.0, 0.0], [0.0, 1.0]],
                    expected=2.0 * GeneralLinear(2).identity,
                )
            ]
            return self.generate_tests(smoke_data)

        def inv_data(self):
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

        def exp_data(self):
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

        def log_data(self):
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

        def projection_and_belongs_data(self):
            smoke_data = [
                dict(n=2, positive_det=False),
                dict(n=2, positive_det=True),
                dict(n=3, positive_det=True),
            ]
            return self.generate_tests(smoke_data)

        def orbit_data(self):
            point = gs.array([[gs.exp(4.0), 0.0], [0.0, gs.exp(2.0)]])
            sqrt = gs.array([[gs.exp(2.0), 0.0], [0.0, gs.exp(1.0)]])
            identity = GeneralLinear(2).identity
            smoke_data = [
                dict(
                    n=3,
                    point=point,
                    base_point=None,
                    expected=gs.array([identity, sqrt, point]),
                ),
                dict(
                    n=3,
                    point=[point, point],
                    base_point=None,
                    expected=[
                        gs.array([identity, sqrt, point]),
                        gs.array([identity, sqrt, point]),
                    ],
                ),
            ]
            return self.generate_tests(smoke_data)

    def test_belongs(self, n, point, expected):
        group = self.cls(n)
        self.assertAllClose(group.belongs(gs.array(point)), gs.array(expected))

    def test_compose(self, n, mat1, mat2, expected):
        group = self.cls(n)
        self.assertAllClose(
            group.compose(gs.array(mat1), gs.array(mat2)), gs.array(expected)
        )

    def test_inv(self, n, mat, expected):
        group = self.cls(n)
        self.assertAllClose(group.inverse(gs.array(mat), gs.array(expected)))

    def test_exp(self, n, tangent_vec, base_point, expected):
        group = self.cls(n)
        expected = gs.cast(expected, gs.float64)
        tangent_vec = gs.cast(gs.array(tangent_vec), gs.float64)
        base_point = (
            None if base_point is None else gs.cast(gs.array(base_point), gs.float64)
        )
        self.assertAllClose(group.exp(tangent_vec, base_point), gs.array(expected))

    def test_log(self, n, point, base_point, expected):
        group = self.cls(n)
        expected = gs.cast(gs.array(expected), gs.float64)
        point = gs.cast(gs.array(point), gs.float64)
        base_point = (
            None if base_point is None else gs.cast(gs.array(base_point), gs.float64)
        )
        self.assertAllClose(group.log(point, base_point), expected)

    def test_projection_and_belongs(self, n, positive_det, n_samples):
        group = self.cls(n, positive_det)
        shape = (n_samples, n, n)
        result = helper.test_projection_and_belongs(group, shape)
        self.assertAllClose(gs.all(result), gs.array(True))

    def test_orbit(self, n, point, base_point, time, expected):
        group = self.cls(n)
        result = group.orbit(gs.array(point), gs.array(base_point))(time)
        self.assertAllClose(result, gs.array(expected))
