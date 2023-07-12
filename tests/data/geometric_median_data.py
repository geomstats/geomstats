import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperboloid import Hyperboloid
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import (
    SPDAffineMetric,
    SPDEuclideanMetric,
    SPDLogEuclideanMetric,
    SPDMatrices,
)
from geomstats.learning.geometric_median import GeometricMedian
from tests.data_generation import TestData

EPSILON = 10e-6


class GeometricMedianTestData(TestData):
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

        smoke_data = [
            dict(estimator=estimator_0, X=X_0, expected=expected_0),
            dict(estimator=estimator_1, X=X_1, expected=expected_1),
            dict(estimator=estimator_2, X=X_2, expected=expected_2),
        ]

        return self.generate_tests(smoke_data)

    def fit_sanity_test_data(self):
        spd_matrices = SPDMatrices(4, equip=False)
        spd_matrices.equip_with_metric(SPDLogEuclideanMetric)
        smoke_data = [
            dict(
                estimator=GeometricMedian(Euclidean(2)),
            ),
            dict(
                estimator=GeometricMedian(Hyperboloid(3)),
            ),
            dict(
                estimator=GeometricMedian(Hypersphere(4)),
            ),
            dict(
                estimator=GeometricMedian(spd_matrices),
            ),
        ]

        return self.generate_tests(smoke_data)
