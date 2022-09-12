import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import (
    SPDMatrices, SPDMetricAffine, SPDMetricEuclidean
    )
from geomstats.learning.geometric_median import GeometricMedian
from tests.data_generation import TestData

EPSILON = 10e-10


class GeometricMedianTestData(TestData):

    def fit_test_data(self):
        estimator_0 = GeometricMedian(SPDMetricEuclidean(n=1))
        X_0 = gs.array([
            [[1.0-EPSILON]],
            [[1.0+EPSILON]],
            [[10.0]]
        ])
        expected_0 = gs.array([[1.0]])

        estimator_1 = GeometricMedian(SPDMetricAffine(n=2))
        X_1 = gs.array([
            [[1.0-EPSILON, 0.0], [0.0, 1.0+EPSILON]],
            [[1.0+EPSILON, 0.0], [0.0, 1.0-EPSILON]],
            [[1.0, 0.0+EPSILON], [0.0-EPSILON, 1.0]],
            [[1.0, 0.0-EPSILON], [0.0+EPSILON, 1.0]],
            [[10.0, 0.0], [0.0, 10.0]]
        ])
        expected_1 = gs.array([[1.0, 0.0], [0.0, 1.0]])

        smoke_data = [
            dict(estimator=estimator_0, X=X_0, expected=expected_0),
            dict(estimator=estimator_1, X=X_1, expected=expected_1),
        ]

        return self.generate_tests(smoke_data)

    def fit_sanity_test_data(self):
        n = 4
        estimator_0 = GeometricMedian(SPDMetricAffine(n))
        space_0 = SPDMatrices(n)

        space_1 = Hypersphere(2)
        estimator_1 = GeometricMedian(space_1.metric)

        smoke_data = [
            dict(estimator=estimator_0, space=space_0),
            dict(estimator=estimator_1, space=space_1),
        ]

        return self.generate_tests(smoke_data)
