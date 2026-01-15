import geomstats.backend as gs

from ._base import BaseEstimatorTestData, TestData


class PairwiseDistsTestData(TestData):
    def dists_among_selves_test_data(self):
        return self.generate_random_data()

    def symmetric_test_data(self):
        return self.generate_random_data(exclude_single=True)

    def general_test_data(self):
        return self.generate_random_data(exclude_single=True)


class EyePairwiseDistsTestData(TestData):
    def __init__(self, dim, n):
        self.dim = dim
        self.n = n
        super().__init__()

    def dists_test_data(self):
        data = [
            dict(
                points=i * gs.eye(self.dim),
                expected=((2 * (i**2)) ** 0.5) * (gs.ones(self.dim) - gs.eye(self.dim)),
            )
            for i in range(1, self.n + 1)
        ]
        return self.generate_tests(data)


class MDSTestData(BaseEstimatorTestData):
    def minimal_fit_test_data(self):
        return self.generate_random_data()

    def minimal_fit_transform_test_data(self):
        return self.generate_random_data()
