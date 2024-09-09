import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import LevelSetTestData, MatrixVectorSpaceTestData


class LowerTriangularMatricesTestData(MatrixVectorSpaceTestData):
    pass


class LowerTriangularMatrices2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 0.0], [-1.0, 3.0]]), expected=True),
            dict(point=gs.array([[1.0, -1.0], [-1.0, 3.0]]), expected=False),
            dict(
                point=gs.array(
                    [
                        [[1.0, 0], [0, 1.0]],
                        [[1.0, 2.0], [2.0, 1.0]],
                        [[-1.0, 0.0], [1.0, 1.0]],
                        [[0.0, 0.0], [1.0, 1.0]],
                    ]
                ),
                expected=gs.array([True, False, True, True]),
            ),
        ]

        return self.generate_tests(data)

    def projection_test_data(self):
        data = [
            dict(
                point=gs.array([[2.0, 1.0], [1.0, 2.0]]),
                expected=gs.array([[2.0, 0.0], [1.0, 2.0]]),
            ),
            dict(
                point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0, 0.0], [0.0, 1.0]]),
            ),
        ]
        return self.generate_tests(data)

    def basis_test_data(self):
        data = [
            dict(
                expected=gs.array(
                    [
                        [[1.0, 0.0], [0.0, 0.0]],
                        [[0.0, 0.0], [1.0, 0.0]],
                        [[0.0, 0.0], [0.0, 1.0]],
                    ]
                ),
            )
        ]
        return self.generate_tests(data)


class LowerTriangularMatrices3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array(
                    [
                        [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        [[-1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    ]
                ),
                expected=[False, True, True, True],
            ),
            dict(point=gs.array([[1.0, 0.0], [-1.0, 3.0]]), expected=gs.array(False)),
        ]
        return self.generate_tests(data)

    def basis_representation_test_data(self):
        data = [
            dict(
                matrix_representation=gs.array(
                    [[1.0, 0.0, 0.0], [0.6, 7.0, 0.0], [-3.0, 0.0, 8.0]]
                ),
                expected=gs.array([1.0, 0.6, 7.0, -3.0, 0.0, 8.0]),
            ),
        ]
        return self.generate_tests(data)


class StrictlyLowerTriangularMatricesTestData(
    LevelSetTestData, MatrixVectorSpaceTestData
):
    pass
