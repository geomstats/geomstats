from polpo.testing.data import CaseData

import geomstats.backend as gs

from .base import LevelSetTestData, MatrixVectorSpaceTestData


class SymmetricMatrices1TestData(CaseData):
    def basis_test_data(self):
        smoke_data = [
            dict(expected=gs.array([[[1.0]]])),
        ]
        return smoke_data

    def basis_representation_test_data(self):
        data = [dict(matrix_representation=gs.array([[1.0]]), expected=gs.array([1.0]))]

        return data

    def matrix_representation_test_data(self):
        data = [
            dict(basis_representation=gs.array([1.0]), expected=gs.array([[1.0]])),
        ]

        return data


class SymmetricMatrices2TestData(CaseData):
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

        return data

    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 2.0], [2.0, 1.0]]), expected=gs.array(True)),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=gs.array(False)),
            dict(
                point=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=gs.array([True, False]),
            ),
        ]
        return data


class SymmetricMatrices3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=gs.array(True),
            ),
        ]

        return data

    def basis_representation_test_data(self):
        data = [
            dict(
                matrix_representation=gs.array(
                    [[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]
                ),
                expected=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
        ]
        return data

    def matrix_representation_test_data(self):
        data = [
            dict(
                basis_representation=gs.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                expected=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
            ),
        ]
        return data


class SymmetricHollowMatricesTestData(LevelSetTestData, MatrixVectorSpaceTestData):
    pass


class NullRowSumsSymmetricMatricesTestData(LevelSetTestData, MatrixVectorSpaceTestData):
    pass
