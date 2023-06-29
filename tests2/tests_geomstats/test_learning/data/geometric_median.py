import random

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.geometric_median import GeometricMedian
from geomstats.test.data import TestData

EPSILON = 10e-6


class GeometricMedianTestData(TestData):
    def estimate_belongs_test_data(self):
        return self.generate_tests([dict(n_points=random.randint(2, 10))])


class GeometricMedianFitTestData(TestData):
    def fit_test_data(self):
        estimator_0 = GeometricMedian(Euclidean(1))
        X_0 = gs.array(
            [
                [1.0 - 2 * EPSILON],
                [1.0 - EPSILON],
                [1.0 + EPSILON],
                [1.0 + 2 * EPSILON],
                [-100.0],
            ]
        )
        expected_0 = gs.array([1.0])

        space = SPDMatrices(1, equip=False)
        space.equip_with_metric(SPDEuclideanMetric)
        estimator_1 = GeometricMedian(space)
        X_1 = gs.array(
            [
                [[1.0 - 2 * EPSILON]],
                [[1.0 - EPSILON]],
                [[1.0 + EPSILON]],
                [[1.0 + 2 * EPSILON]],
                [[10.0]],
            ]
        )
        expected_1 = gs.array([[1.0]])

        space = SPDMatrices(2, equip=False)
        space.equip_with_metric(SPDAffineMetric)
        estimator_2 = GeometricMedian(space)
        X_2 = gs.array(
            [
                [[1.0 + EPSILON, 0.0], [0.0, 1.0 + EPSILON]],
                [[1.0 - EPSILON, 0.0], [0.0, 1.0 - EPSILON]],
                [[1.0, 0.0 + EPSILON], [0.0 + EPSILON, 1.0]],
                [[1.0, 0.0 - EPSILON], [0.0 - EPSILON, 1.0]],
                [[10.0, 0.0], [0.0, 10.0]],
            ]
        )
        expected_2 = gs.array([[1.0, 0.0], [0.0, 1.0]])

        atol = 1e-5
        data = [
            dict(estimator=estimator_0, X=X_0, expected=expected_0, atol=atol),
            dict(estimator=estimator_1, X=X_1, expected=expected_1, atol=atol),
            dict(estimator=estimator_2, X=X_2, expected=expected_2, atol=atol),
        ]

        return self.generate_tests(data)
