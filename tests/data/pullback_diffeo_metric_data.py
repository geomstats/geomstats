import geomstats.backend as gs
from tests.data_generation import TestData


class PullbackDiffeoCircleMetricTestData(TestData):
    def diffeomorphism_is_reciprocal_test_data(self):
        smoke_data = [
            dict(point=gs.array([1, 0])),
        ]
        return self.generate_tests(smoke_data)

    def tangent_diffeomorphism_is_reciprocal_test_data(self):
        smoke_data = [
            dict(point=gs.array([1, 0]), tangent_vector=gs.array([0, 2])),
        ]
        return self.generate_tests(smoke_data)

    def matrix_innerproduct_and_embedded_innerproduct_coincide_test_data(self):
        smoke_data = []
        return self.generate_tests(smoke_data)
