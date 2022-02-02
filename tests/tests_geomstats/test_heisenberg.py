"""Unit tests for the 3D heisenberg group in vector representation."""

import geomstats.backend as gs
from geomstats.geometry.heisenberg import HeisenbergVectors
from tests.conftest import Parametrizer, TestCase, TestData


class TestHeisenbergVectors(TestCase, metaclass=Parametrizer):
    cls = HeisenbergVectors

    class TestDataHeisenbergVectors(TestData):
        def dimension_data(self):
            smoke_data = [dict(expected=3)]
            return self.generate_tests(smoke_data)

        def belongs_data(self):
            smoke_data = [
                dict(point=[1.0, 2.0, 3.0, 4], expected=False),
                dict(
                    point=[[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]],
                    expected=[False, False],
                ),
            ]
            return self.generate_tests(smoke_data)

        def is_tangent_data(self):
            smoke_data = [
                dict(
                    vec=[[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
                    expected=[False, False],
                )
            ]
            return self.generate_tests(smoke_data)

        def jacobian_translation_data(self):
            smoke_data = [
                dict(
                    vec=[[1.0, -10.0, 0.2], [-2.0, 100.0, 0.5]],
                    expected=[
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.5, 1.0]],
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-50.0, -1.0, 1.0]],
                    ],
                )
            ]
            return self.generate_tests(smoke_data)

    def test_dimension(self, expected):
        self.assertAllClose(self.cls().dim, expected)

    def test_jacobian_translation(self, vec, expected):
        self.assertAllClose(
            self.cls().jacobian_translation(gs.array(vec), gs.array(expected))
        )

    def test_random_point_belongs(self, n_samples, bound):
        self.assertAllClose(gs.all(self.cls().random_point(n_samples, bound)), True)
