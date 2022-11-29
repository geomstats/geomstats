import geomstats.backend as gs
from geomstats.test.data import TestData
from tests2.data.base_data import MatrixLieAlgebraTestData


def algebra_useful_matrix(theta, elem_33=0.0):
    return gs.array([[0.0, -theta, 2.0], [theta, 0.0, 3.0], [0.0, 0.0, elem_33]])


class SpecialEuclideanMatrixLieAlgebraTestData(MatrixLieAlgebraTestData):
    pass


class SpecialEuclideanMatrixLieAlgebra2TestData(TestData):
    def belongs_test_data(self):
        theta = gs.pi / 3
        data = [
            dict(
                point=algebra_useful_matrix(theta, elem_33=0.0), expected=gs.array(True)
            ),
            dict(
                point=algebra_useful_matrix(theta, elem_33=1.0),
                expected=gs.array(False),
            ),
            dict(
                point=gs.stack(
                    [
                        algebra_useful_matrix(theta, elem_33=0.0),
                        algebra_useful_matrix(theta, elem_33=1.0),
                    ]
                ),
                expected=gs.array([True, False]),
            ),
        ]
        return self.generate_tests(data)
