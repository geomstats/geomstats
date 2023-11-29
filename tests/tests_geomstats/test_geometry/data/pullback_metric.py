import geomstats.backend as gs
from geomstats.test.data import TestData

from .riemannian_metric import (
    RiemannianMetricComparisonTestData,
    RiemannianMetricTestData,
)


def _expected_jacobian_circle_immersion(point):
    return gs.stack(
        [
            -gs.sin(point),
            gs.cos(point),
        ],
        axis=-2,
    )


def _expected_jacobian_sphere_immersion(point):
    theta = point[..., 0]
    phi = point[..., 1]
    jacobian = gs.array(
        [
            [gs.cos(phi) * gs.cos(theta), -gs.sin(phi) * gs.sin(theta)],
            [gs.sin(phi) * gs.cos(theta), gs.cos(phi) * gs.sin(theta)],
            [-gs.sin(theta), 0.0],
        ]
    )
    return jacobian


def _expected_hessian_sphere_immersion(point):
    theta = point[..., 0]
    phi = point[..., 1]
    hessian_immersion_x = gs.array(
        [
            [-gs.sin(theta) * gs.cos(phi), -gs.cos(theta) * gs.sin(phi)],
            [-gs.cos(theta) * gs.sin(phi), -gs.sin(theta) * gs.cos(phi)],
        ]
    )
    hessian_immersion_y = gs.array(
        [
            [-gs.sin(theta) * gs.sin(phi), gs.cos(theta) * gs.cos(phi)],
            [gs.cos(theta) * gs.cos(phi), -gs.sin(theta) * gs.sin(phi)],
        ]
    )
    hessian_immersion_z = gs.array([[-gs.cos(theta), 0.0], [0.0, 0.0]])
    hessian_immersion = gs.stack(
        [hessian_immersion_x, hessian_immersion_y, hessian_immersion_z], axis=0
    )
    return hessian_immersion


def _expected_circle_metric_matrix(point):
    mat = gs.array([[1.0]])
    return mat


def _expected_sphere_metric_matrix(point):
    theta = point[..., 0]
    mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** 2]])
    return mat


def _expected_inverse_circle_metric_matrix(point):
    mat = gs.array([[1.0]])
    return mat


def _expected_inverse_sphere_metric_matrix(point):
    theta = point[..., 0]
    mat = gs.array([[1.0, 0.0], [0.0, gs.sin(theta) ** (-2)]])
    return mat


def _expected_circle_metric_second_fundamental_form(base_point):
    return gs.array(
        [
            [-gs.cos(base_point)],
            [-gs.sin(base_point)],
        ]
    )


def _expected_sphere_metric_second_fundamental_form(base_point, radius=1.0):
    theta, phi = base_point

    expected_11 = gs.array(
        [
            -radius * gs.sin(theta) * gs.cos(phi),
            -radius * gs.sin(theta) * gs.sin(phi),
            -radius * gs.cos(theta),
        ]
    )
    expected_22 = gs.array(
        [
            -radius * gs.sin(theta) ** 2 * gs.sin(theta) * gs.cos(phi),
            -radius * gs.sin(theta) ** 2 * gs.sin(theta) * gs.sin(phi),
            -radius * gs.sin(theta) ** 2 * gs.cos(theta),
        ]
    )
    return expected_11, expected_22


def _expected_sphere_inner_product_derivativ_matrix(base_point):
    theta = base_point[0]

    # derivative with respect to theta
    expected_1 = gs.array([[0, 0], [0, 2 * gs.cos(theta) * gs.sin(theta)]])
    # derivative with respect to phi
    expected_2 = gs.zeros((2, 2))

    return expected_1, expected_2


class CircleIntrinsicTestData(TestData):
    fail_for_autodiff_exceptions = False

    def tangent_immersion_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([1.0]),
                base_point=gs.array([0.0]),
                expected=gs.array([0.0, 1.0]),
            ),
        ]
        return self.generate_tests(data)

    def jacobian_immersion_test_data(self):
        base_points = [gs.array([0.0]), gs.array([0.2]), gs.array([4.0])]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_jacobian_circle_immersion(base_point),
                ),
            )

        return self.generate_tests(data)


class SphereIntrinsicTestData(TestData):
    fail_for_autodiff_exceptions = False

    def immersion_test_data(self):
        data = [
            dict(
                point=gs.array([0.0, 0.0]),
                expected=gs.array([0.0, 0.0, 1.0]),
            ),
            dict(
                point=gs.array([gs.pi, 0.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                point=gs.array([gs.pi / 2.0, gs.pi]),
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
        ]
        return self.generate_tests(data)

    def tangent_immersion_test_data(self):
        data = [
            dict(
                tangent_vec=gs.array([1.0, 0.0]),
                base_point=gs.array([gs.pi / 2.0, gs.pi / 2.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                tangent_vec=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, gs.pi / 2.0]),
                expected=gs.array([-1.0, 0.0, 0.0]),
            ),
            dict(
                tangent_vec=gs.array([1.0, 0.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
                expected=gs.array([0.0, 0.0, -1.0]),
            ),
            dict(
                tangent_vec=gs.array([0.0, 1.0]),
                base_point=gs.array([gs.pi / 2.0, 0.0]),
                expected=gs.array([0.0, 1.0, 0.0]),
            ),
        ]
        return self.generate_tests(data)

    def jacobian_immersion_test_data(self):
        base_points = [
            gs.array([0.0, 0.0]),
            gs.array([0.22, 0.1]),
            gs.array([0.1, 0.88]),
        ]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_jacobian_sphere_immersion(base_point),
                )
            )
        return self.generate_tests(data)

    def hessian_immersion_test_data(self):
        base_points = [
            gs.array([0.0, 0.0]),
            gs.array([0.22, 0.1]),
            gs.array([0.1, 0.88]),
        ]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_hessian_sphere_immersion(base_point),
                )
            )
        return self.generate_tests(data)


class CircleIntrinsicMetricTestData(TestData):
    fail_for_autodiff_exceptions = False

    def metric_matrix_test_data(self):
        base_points = [gs.array([0.0]), gs.array([1.0]), gs.array([4.0])]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_circle_metric_matrix(base_point),
                )
            )

        return self.generate_tests(data)

    def cometric_matrix_test_data(self):
        base_points = [gs.array([0.6]), gs.array([0.8])]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_inverse_circle_metric_matrix(base_point),
                )
            )

        return self.generate_tests(data)

    def second_fundamental_form_test_data(self):
        base_points = [gs.array([0.22]), gs.array([0.88])]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_circle_metric_second_fundamental_form(
                        base_point
                    ),
                )
            )

        return self.generate_tests(data)

    def mean_curvature_vector_norm_test_data(self):
        radius = 1.0
        expected = gs.array(1 / radius)
        data = [
            dict(base_point=gs.array([0.1]), expected=expected),
            dict(base_point=gs.array([0.88]), expected=expected),
        ]
        return self.generate_tests(data)

    def christoffels_test_data(self):
        data = [
            dict(base_point=gs.array([0.1]), expected=gs.zeros((1, 1, 1))),
            dict(base_point=gs.array([0.88]), expected=gs.zeros((1, 1, 1))),
        ]
        return self.generate_tests(data)


class SphereIntrinsicMetricTestData(TestData):
    fail_for_autodiff_exceptions = False

    def metric_matrix_test_data(self):
        base_points = [gs.array([0.0, 0.0]), gs.array([1.0, 1.0]), gs.array([0.3, 0.8])]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_sphere_metric_matrix(base_point),
                )
            )

        return self.generate_tests(data)

    def cometric_matrix_test_data(self):
        base_points = [gs.array([0.6, -1.0]), gs.array([0.8, -0.8])]
        data = []
        for base_point in base_points:
            data.append(
                dict(
                    base_point=base_point,
                    expected=_expected_inverse_sphere_metric_matrix(base_point),
                )
            )

        return self.generate_tests(data)

    def second_fundamental_form_test_data(self):
        base_points = [gs.array([0.22, 0.1]), gs.array([0.1, 0.88])]
        data = []
        for base_point in base_points:
            expected_11, expected_22 = _expected_sphere_metric_second_fundamental_form(
                base_point
            )
            data.append(
                dict(
                    base_point=base_point,
                    expected_11=expected_11,
                    expected_22=expected_22,
                )
            )
        return self.generate_tests(data)

    def mean_curvature_vector_norm_test_data(self):
        radius = 1.0
        expected = gs.array(2 / radius)
        data = [
            dict(base_point=gs.array([0.22, 0.1]), expected=expected),
            dict(base_point=gs.array([0.1, 0.88]), expected=expected),
        ]
        return self.generate_tests(data)

    def inner_product_derivative_matrix_test_data(self):
        base_points = [gs.array([0.6, -1.0]), gs.array([0.8, -0.8])]
        data = []
        for base_point in base_points:
            expected_1, expected_2 = _expected_sphere_inner_product_derivativ_matrix(
                base_point
            )
            data.append(
                dict(
                    base_point=base_point,
                    expected_1=expected_1,
                    expected_2=expected_2,
                )
            )
        return self.generate_tests(data)


class CircleAsSO2PullbackDiffeoMetricCmpTestData(RiemannianMetricComparisonTestData):
    fail_for_autodiff_exceptions = False
    fail_for_not_implemented_errors = False


class PullbackDiffeoMetricTestData(RiemannianMetricTestData):
    pass
