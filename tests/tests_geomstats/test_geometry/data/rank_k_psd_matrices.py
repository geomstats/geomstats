import geomstats.backend as gs
from geomstats.test.data import TestData

from .fiber_bundle import FiberBundleTestData
from .manifold import ManifoldTestData
from .mixins import ProjectionMixinsTestData
from .quotient_metric import QuotientMetricTestData


class RankKPSDMatricesTestData(ProjectionMixinsTestData, ManifoldTestData):
    xfails = (
        "to_tangent_is_tangent",
        "random_tangent_vec_is_tangent",
        "projection_belongs",
    )


class RankKPSDMatrices32TestData(TestData):
    def belongs_test_data(self):
        data = [
            dict(
                point=gs.array(
                    [
                        [0.8369314, -0.7342977, 1.0402943],
                        [0.04035992, -0.7218659, 1.0794858],
                        [0.9032698, -0.73601735, -0.36105633],
                    ]
                ),
                expected=False,
            ),
            dict(
                point=gs.array([[1.0, 1.0, 0], [1.0, 4.0, 0], [0, 0, 0]]),
                expected=True,
            ),
        ]
        return self.generate_tests(data)


class BuresWassersteinBundleTestData(FiberBundleTestData):
    fail_for_not_implemented_errors = False
    skips = (
        "horizontal_lift_vec",
        "horizontal_lift_is_horizontal",
        "integrability_tensor_derivative_vec",
    )

    xfails = ("tangent_riemannian_submersion_after_horizontal_lift",)


class PSDBuresWassersteinMetricTestData(QuotientMetricTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False

    skips = (
        "curvature_derivative_vec",
        "directional_curvature_derivative_vec",
    )

    xfails = ("log_after_exp",)


class PSD22BuresWassersteinMetricTestData(TestData):
    def exp_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[4.0, 0.0], [0.0, 4.0]]),
            )
        ]
        return self.generate_tests(data)

    def log_test_data(self):
        data = [
            dict(
                point=gs.array([[4.0, 0.0], [0.0, 4.0]]),
                base_point=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                expected=gs.array([[2.0, 0.0], [0.0, 2.0]]),
            )
        ]
        return self.generate_tests(data)

    def squared_dist_test_data(self):
        data = [
            dict(
                point_a=gs.array([[1.0, 0.0], [0.0, 1.0]]),
                point_b=gs.array([[2.0, 0.0], [0.0, 2.0]]),
                expected=gs.array(2 + 4 - (2 * 2 * 2**0.5)),
            )
        ]
        return self.generate_tests(data)


class PSD33BuresWassersteinMetricTestData(TestData):
    def inner_product_test_data(self):
        data = [
            dict(
                tangent_vec_a=gs.array(
                    [[2.0, 1.0, 1.0], [1.0, 0.5, 0.5], [1.0, 0.5, 0.5]]
                ),
                tangent_vec_b=gs.array(
                    [[1.0, 2.0, 4.0], [2.0, 3.0, 8.0], [4.0, 8.0, 5.0]]
                ),
                base_point=gs.array(
                    [[1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.5, 1.5]]
                ),
                expected=gs.array(4.0),
            )
        ]
        return self.generate_tests(data)
