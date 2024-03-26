import geomstats.backend as gs
from geomstats.test.data import TestData

from .riemannian_metric import (
    RiemannianMetricCmpWithTransformTestData,
)


class HalfSpaceToBall2TestData(TestData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array([0.0, 1.0]),
                expected=gs.zeros(2),
            ),
            dict(
                base_point=gs.array([[0.0, 1.0], [0.0, 2.0]]),
                expected=gs.array([[0.0, 0.0], [0.0, 1.0 / 3.0]]),
            ),
        ]
        return self.generate_tests(data)


class ExtrinsicToBall3TestData(TestData):
    def diffeomorphism_test_data(self):
        data = [
            dict(
                base_point=gs.array([2.0, 1.0, gs.sqrt(2)]),
                expected=gs.array([1.0 / 3.0, gs.sqrt(2) / 3.0]),
            ),
        ]
        return self.generate_tests(data)

    def tangent_diffeomorphism_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([1.0, 1.0, 1.0]),
                base_point=gs.array([1.0, 0.0, 0.0]),
                expected=gs.array([1.0 / 2.0, 1.0 / 2.0]),
            )
        ]
        return self.generate_tests(data)


class HyperbolicCmpWithTransformTestData(RiemannianMetricCmpWithTransformTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False
