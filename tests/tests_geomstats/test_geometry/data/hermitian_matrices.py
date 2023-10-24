import geomstats.backend as gs
from geomstats.test.data import TestData

from .base import ComplexVectorSpaceTestData, MatrixVectorSpaceMixinsTestData


class HermitianMatricesTestData(
    MatrixVectorSpaceMixinsTestData, ComplexVectorSpaceTestData
):
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
    def to_vector_test_data(self):
        data = [
            dict(
                point=gs.array(
                    [[1.0, 2.0, 3.0 + 1.0j], [2.0, 4.0, 5.0], [3.0 - 1.0j, 5.0, 6.0]]
                ),
                expected=gs.array([1.0, 4.0, 6.0, 2.0, 3.0, 5.0, 0.0, 1.0, 0.0]),
            ),
        ]
        return self.generate_tests(data)

    def from_vector_test_data(self):
        data = [
            dict(
                vec=gs.array([1.0, 4.0, 6.0, 2.0, 3.0, 5.0, 0.0, 1.0, 0.0]),
                expected=gs.array(
                    [[1.0, 2.0, 3.0 + 1j], [2.0, 4.0, 5.0], [3.0 - 1j, 5.0, 6.0]]
                ),
            ),
        ]
        return self.generate_tests(data)


class HermitianMatricesOpsTestData(TestData):
    def expm_test_data(self):
        data = [
            dict(
                mat=gs.array([[0.0, 1.0j], [-1.0j, 0.0]]),
                expected=gs.array(
                    [[1.54308063, 1.17520119j], [-1.17520119j, 1.54308063]]
                ),
            )
        ]
        return self.generate_tests(data)

    def powerm_test_data(self):
        data = [
            dict(
                mat=gs.array([[0.0, 1.0j], [-1.0j, 0.0]]),
                power=1,
                expected=gs.array([[0.0, 1.0j], [-1.0j, 0.0]]),
            ),
            dict(
                mat=gs.array([[0.0, 1.0j], [-1.0j, 0.0]]),
                power=2,
                expected=gs.array(
                    [
                        [
                            1.0 + 0j,
                            0.0,
                        ],
                        [0.0, 1.0],
                    ]
                ),
            ),
        ]

        return self.generate_tests(data)
