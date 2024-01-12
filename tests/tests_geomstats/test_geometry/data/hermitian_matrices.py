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
