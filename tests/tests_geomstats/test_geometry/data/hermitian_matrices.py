import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import ComplexMatrixVectorSpaceTestData


class HermitianMatricesTestData(ComplexMatrixVectorSpaceTestData):
    pass


class HermitianMatrices2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[1.0, 2.0 + 1j], [2.0 - 1j, 1.0]]), expected=True),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                point=gs.array(
                    [[[1.0, 1j], [-1j, 1.0]], [[1.0 + 1j, -1.0], [0.0, 1.0]]]
                ),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(data)


class HermitianMatrices3TestData(TestData):
    def basis_representation_test_data(self):
        data = [
            dict(
                matrix_representation=gs.array(
                    [[1.0, 2.0, 3.0 + 1.0j], [2.0, 4.0, 5.0], [3.0 - 1.0j, 5.0, 6.0]]
                ),
                expected=gs.array([1.0, 4.0, 6.0, 2.0, 3.0, 5.0, 0.0, 1.0, 0.0]),
            ),
        ]
        return self.generate_tests(data)

    def matrix_representation_test_data(self):
        data = [
            dict(
                basis_representation=gs.array(
                    [1.0, 4.0, 6.0, 2.0, 3.0, 5.0, 0.0, 1.0, 0.0]
                ),
                expected=gs.array(
                    [[1.0, 2.0, 3.0 + 1j], [2.0, 4.0, 5.0], [3.0 - 1j, 5.0, 6.0]]
                ),
            ),
        ]
        return self.generate_tests(data)
