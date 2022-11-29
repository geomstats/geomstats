import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.data.base_data import MatrixLieAlgebraTestData


class SkewSymmetricMatricesTestData(MatrixLieAlgebraTestData):
    def baker_campbell_hausdorff_with_basis_test_data(self):
        return self.generate_tests([dict()])


class SkewSymmetricMatrices2TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[0.0, -1.0], [1.0, 0.0]]), expected=gs.array(True)),
        ]
        return self.generate_tests(data)


class SkewSymmetricMatrices3TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(point=gs.array([[0.0, -1.0], [1.0, 0.0]]), expected=gs.array(False)),
        ]
        return self.generate_tests(data)
