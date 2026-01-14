import geomstats.backend as gs
from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData


class MDSTestData(BaseEstimatorTestData):
    N_SAMPLES = 4

    def dissimilarity_against_self_test_data(self):
        return self.generate_random_data()


class MDSEuclideanTestData(TestData):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def dissimilarity_matrix_test_data(self):
        data = [
            dict(
                X=gs.eye(self.n), expected=(2**0.5) * (gs.ones(self.n) - gs.eye(self.n))
            ),
            dict(
                X=2 * gs.eye(self.n),
                expected=(2**0.5) * (gs.ones(self.n) - gs.eye(self.n)),
            ),
        ]
        return self.generate_tests(data)

    def dissimilarity_matrix2_test_data(self):
        data = [
            dict(
                X=gs.eye(self.n), expected=(2**0.5) * (gs.ones(self.n) - gs.eye(self.n))
            ),
            dict(
                X=2 * gs.eye(self.n),
                expected=(2**0.5) * (gs.ones(self.n) - gs.eye(self.n)),
            ),
        ]
        return self.generate_tests(data)


class MDSSPDTestData(TestData):
    def __init__(self, n):
        self.n = n
        super().__init__()

    def dissimilarity_matrix_test_data(self):
        data = [
            dict(
                X=gs.array([gs.eye(self.n)] * self.n),
                expected=(gs.zeros((self.n, self.n))),
            ),
        ]
        return self.generate_tests(data)
