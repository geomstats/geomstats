from polpo.testing.data import CaseData

import geomstats.backend as gs

from .lie_algebra import MatrixLieAlgebraTestData


class SkewSymmetricMatricesTestData(MatrixLieAlgebraTestData):
    def baker_campbell_hausdorff_with_basis_test_data(self):
        return [dict()]


class SkewSymmetricMatrices2TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[0.0, -1.0], [1.0, 0.0]]), expected=gs.array(True)),
        ]
        return data

    def matrix_representation_test_data(self):
        data = [
            dict(
                basis_representation=gs.array([0.9]),
                expected=gs.array([[0.0, -0.9], [0.9, 0.0]]),
            )
        ]
        return data


class SkewSymmetricMatrices3TestData(CaseData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[0.0, -1.0], [1.0, 0.0]]), expected=gs.array(False)),
        ]
        return data
