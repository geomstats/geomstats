import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import LevelSetTestData, MatrixVectorSpaceTestData
from .euclidean import EuclideanMetricTestData


class SymmetricMatricesTestData(MatrixVectorSpaceTestData):
    pass


class SymmetricMatrices1TestData(TestData):
    def basis_test_data(self):
        smoke_data = [
            dict(expected=gs.array([[[1.0]]])),
        ]
        return self.generate_tests(smoke_data)

    def basis_representation_test_data(self):
        data = [dict(matrix_representation=gs.array([[1.0]]), expected=gs.array([1.0]))]

        return self.generate_tests(data)

    def matrix_representation_test_data(self):
        data = [
            dict(basis_representation=gs.array([1.0]), expected=gs.array([[1.0]])),
        ]

        return self.generate_tests(data)


class SymmetricMatrices2TestData(TestData):
    def basis_test_data(self):
        data = [
            dict(
                expected=gs.array(
                    [
                        [[1.0, 0.0], [0, 0]],
                        [[0, 1.0], [1.0, 0]],
                        [[0, 0.0], [0, 1.0]],
                    ]
                ),
            ),
        ]

        return self.generate_tests(data)

    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 2.0], [2.0, 1.0]]), expected=gs.array(True)),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=gs.array(False)),
            dict(
                point=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(data)


class SymmetricMatrices3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=gs.array(True),
            ),
        ]

        return self.generate_tests(data)

    def basis_representation_test_data(self):
        data = [
            dict(
                matrix_representation=gs.array(
                    [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
                ),
                expected=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
        ]
        return self.generate_tests(data)

    def matrix_representation_test_data(self):
        data = [
            dict(
                basis_representation=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                expected=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
            ),
        ]
        return self.generate_tests(data)


class SymmetricHollowMatricesTestData(LevelSetTestData, MatrixVectorSpaceTestData):
    pass


class HollowMatricesPermutationInvariantMetricTestData(EuclideanMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False


class NullRowSumSymmetricMatricesTestData(LevelSetTestData, MatrixVectorSpaceTestData):
    pass


class NullRowSumPermutationInvariantMetricTestData(EuclideanMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False
