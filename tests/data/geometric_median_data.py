import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.minkowski import Minkowski
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricAffine,
    SPDMetricEuclidean,
)
from geomstats.learning.geometric_median import GeometricMedian
from tests.data_generation import TestData

EPSILON = 10e-6


class GeometricMedianTestData(TestData):
    def fit_test_data(self):
        estimator_0 = GeometricMedian(SPDMetricEuclidean(n=1))
        X_0 = gs.array(
            [
                [[1.0 - EPSILON]],
                [[1.0 + EPSILON]],
                [[1.0 - 2 * EPSILON]],
                [[1.0 + 2 * EPSILON]],
                [[10.0]],
            ]
        )
        expected_0 = gs.array([[1.0]])

        estimator_1 = GeometricMedian(SPDMetricAffine(n=2))
        X_1 = gs.array(
            [
                [[1.0 + EPSILON, 0.0], [0.0, 1.0 + EPSILON]],
                [[1.0 - EPSILON, 0.0], [0.0, 1.0 - EPSILON]],
                [[1.0, 0.0 + EPSILON], [0.0 + EPSILON, 1.0]],
                [[1.0, 0.0 - EPSILON], [0.0 - EPSILON, 1.0]],
                [[10.0, 0.0], [0.0, 10.0]],
            ]
        )
        expected_1 = gs.array([[1.0, 0.0], [0.0, 1.0]])

        smoke_data = [
            dict(estimator=estimator_0, X=X_0, expected=expected_0),
            dict(estimator=estimator_1, X=X_1, expected=expected_1),
        ]

        return self.generate_tests(smoke_data)

    def fit_sanity_test_data(self):

        smoke_data = [
            dict(
                estimator=GeometricMedian(Euclidean(2).metric),
                space=Euclidean(2),
            ),
            dict(
                estimator=GeometricMedian(Hyperboloid(3).metric),
                space=Hyperboloid(3),
            ),
            dict(
                estimator=GeometricMedian(Minkowski(3).metric),
                space=Minkowski(3),
            ),
            dict(
                estimator=GeometricMedian(SPDMetricAffine(4)),
                space=SPDMatrices(4),
            ),
        ]

        return self.generate_tests(smoke_data)
