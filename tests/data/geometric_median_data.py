import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine
from geomstats.learning.geometric_median import GeometricMedian
from tests.data_generation import TestData


class GeometricMedianTestData(TestData):
    def fit_test_data(self):
        estimator = GeometricMedian(SPDMetricAffine(n=2))
        X = gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        expected = gs.array([[1.0, 0.0], [0.0, 1.0]])

        smoke_data = [dict(estimator=estimator, X=X, expected=expected)]

        return self.generate_tests(smoke_data)

    def fit_sanity_test_data(self):
        n = 4
        estimator_1 = GeometricMedian(SPDMetricAffine(n))
        space_1 = SPDMatrices(n)

        space_2 = Hypersphere(2)
        estimator_2 = GeometricMedian(space_2.metric)

        smoke_data = [
            dict(estimator=estimator_1, space=space_1),
            dict(estimator=estimator_2, space=space_2),
        ]

        return self.generate_tests(smoke_data)
