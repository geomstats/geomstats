import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.data.base_data import MatrixLieAlgebraTestData


class SquareMatricesTestData(MatrixLieAlgebraTestData):
    pass


class SquareMatrices3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.eye(3), expected=gs.array(True)),
            dict(point=gs.ones((3, 3)), expected=gs.array(True)),
            dict(point=gs.ones(3), expected=gs.array(False)),
        ]
        return self.generate_tests(data)
