import geomstats.backend as gs
from geomstats.test.data import TestData

from .complex_matrices import ComplexMatricesMetricTestData
from .complex_riemannian_metric import ComplexRiemannianMetricTestData
from .pullback_metric import PullbackDiffeoMetricTestData


class HPDMatrices2TestData(TestData):
    def belongs_test_data(self):
        smoke_data = [
            dict(point=gs.array([[3.0, -1.0], [-1.0, 3.0]]), expected=True),
            dict(point=gs.array([[3j, -1.0], [-1.0, 3.0]]), expected=False),
            dict(point=gs.array([[1.0, 1.0], [2.0, 1.0]]), expected=False),
            dict(
                point=gs.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, -1.0], [0.0, 1.0]]]),
                expected=[True, False],
            ),
        ]
        return self.generate_tests(smoke_data)

    def projection_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([[1.0 + 0.0j, 0.0], [0.0, 1.0]]),
                expected=gs.array([[1.0 + 0.0j, 0.0], [0.0, 1.0]]),
            ),
            dict(
                point=gs.array([[1.0 + 0.0j, 0.5j], [0.5j, 1.0 + 0.0j]]),
                expected=gs.array([[1.0 + 0.0j, 0.0], [0.0, 1.0]]),
            ),
            dict(
                point=gs.array([[-1.0 + 0.0j, 0.0], [0.0, -2.0]]),
                expected=gs.array([[gs.atol + 0.0j, 0.0], [0.0, gs.atol]]),
            ),
        ]
        return self.generate_tests(smoke_data)


class HPDMatrices3TestData(TestData):
    def belongs_test_data(self):
        smoke_data = [
            dict(
                point=gs.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]]),
                expected=False,
            ),
            dict(
                point=gs.array(
                    [[3.0 + 0j, 0j, 1j], [0j, 4.0 + 0j, 0j], [-0.5j, 0j, 6.0 + 0j]]
                ),
                expected=False,
            ),
        ]
        return self.generate_tests(smoke_data)


class HPDAffineMetricTestData(ComplexRiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False


class HPDBuresWassersteinMetricTestData(ComplexRiemannianMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    tolerances = {
        "dist_is_log_norm": {"atol": 1e-5},
        "dist_point_to_itself_is_zero": {"atol": 1e-3},
    }


class HPDEuclideanMetricTestData(ComplexMatricesMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False

    def exp_domain_vec_test_data(self):
        return self.generate_vec_data()


class HPDLogEuclideanMetricTestData(PullbackDiffeoMetricTestData):
    fail_for_not_implemented_errors = False
    fail_for_autodiff_exceptions = False
