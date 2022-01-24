"""Unit tests for special euclidean group in matrix representation."""


import geomstats.backend as gs
from tests.conftest import Parametrizer, TestCase, TestData


class TestSpecialEuclidean(TestCase, metaclass=Parametrizer):
    class TestDataSpecialEuclidean(TestData):
        def test_belongds_data(self):
            theta = gs.pi / 3

            smoke_data = [
                dict(
                    n=2,
                    mat=[
                        [gs.cos(theta), -gs.sin(theta), 2.0],
                        [gs.sin(theta), gs.cos(theta), 3.0],
                        [0.0, 0.0, 1.0],
                    ],
                    expected=True,
                ),
                dict(
                    n=2,
                    mat=[
                        [gs.cos(theta), -gs.sin(theta), 2.0],
                        [gs.sin(theta), gs.cos(theta), 3.0],
                        [0.0, 0.0, 0.0],
                    ],
                    expected=True,
                ),
            ]
            return self.generate_tests(smoke_data)
