import geomstats.backend as gs
from geomstats.geometry.matrices import Matrices
from geomstats.test.data import TestData

from .fiber_bundle import FiberBundleTestData
from .matrices import RiemannianMetricTestData
from .riemannian_metric import (
    RiemannianMetricComparisonTestData,
)


class Grassmannian32TestData(TestData):
    def belongs_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        p_yz = gs.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p_xz = gs.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        data = [
            dict(point=p_xy, expected=gs.array(True)),
            dict(point=gs.stack([p_yz, p_xz]), expected=gs.array([True, True])),
        ]
        return self.generate_tests(data)


class GrassmannianCanonicalMetricTestData(RiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False
    trials = 3

    xfails = ("log_after_exp",)


class GrassmannianCanonicalMetric32TestData(TestData):
    def exp_test_data(self):
        p_xy = gs.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        p_yz = gs.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p_xz = gs.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        r_y = gs.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        r_z = gs.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        pi_2 = gs.pi / 2

        data = [
            dict(
                tangent_vec=Matrices.bracket(pi_2 * r_y, gs.stack([p_xy, p_yz])),
                base_point=gs.stack([p_xy, p_yz]),
                expected=gs.stack([p_yz, p_xy]),
            ),
            dict(
                tangent_vec=Matrices.bracket(
                    pi_2 * gs.stack([r_y, r_z]), gs.stack([p_xy, p_yz])
                ),
                base_point=gs.stack([p_xy, p_yz]),
                expected=gs.stack([p_yz, p_xz]),
            ),
        ]
        return self.generate_tests(data)


class GrassmannianBundleTestData(FiberBundleTestData):
    fail_for_not_implemented_errors = False


class GrassmannianQuotientMetricCmpTestData(RiemannianMetricComparisonTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False
