import geomstats.backend as gs
from geomstats.test.data import TestData

from ._base import BaseEstimatorTestData


class TangentPCATestData(BaseEstimatorTestData):
    MIN_RANDOM = 5
    MAX_RANDOM = 10

    def fit_inverse_transform_test_data(self):
        return self.generate_random_data()

    def fit_transform_and_transform_after_fit_test_data(self):
        return self.generate_random_data()

    def n_components_test_data(self):
        return self.generate_random_data()

    def n_components_explained_variance_ratio_test_data(self):
        return self.generate_random_data()

    def n_components_mle_test_data(self):
        return self.generate_random_data()


class TangentPCAEuclideanTestData(TestData):
    def n_components_mle_test_data(self):
        data = [
            dict(
                X=gs.array(
                    [
                        [3.0, 0.0, 0.0],
                        [-3.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, -2.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, -1.0],
                    ]
                ),
                expected=1,
            ),
            dict(
                X=gs.array(
                    [
                        [1.0, 2.0, 3.0],
                        [2.0, 4.0, 6.0],
                        [-1.0, -2.0, -3.0],
                        [-2.0, -4.0, -6.0],
                    ]
                ),
                expected=1,
            ),
        ]
        return self.generate_tests(data)
