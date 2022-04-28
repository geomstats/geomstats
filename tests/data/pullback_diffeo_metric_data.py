import geomstats.backend as gs
from tests.data_generation import TestData

RTOL = 1e-5
ATOL = 1e-6


class PullbackDiffeoCircleMetricTestData(TestData):
    def diffeomorphism_is_reciprocal_test_data(self):
        smoke_data = [
            dict(metric_args=[], point=gs.array([1, 0]), rtol=RTOL, atol=ATOL),
        ]
        return self.generate_tests(smoke_data)

    def tangent_diffeomorphism_is_reciprocal_test_data(self):
        smoke_data = [
            dict(
                metric_args=[],
                point=gs.array([1, 0]),
                tangent_vector=gs.array([0, 2]),
                rtol=RTOL,
                atol=ATOL,
            ),
        ]
        return self.generate_tests(smoke_data)

    def matrix_innerproduct_and_embedded_innerproduct_coincide_test_data(self):
        smoke_data = []
        return self.generate_tests(smoke_data)
